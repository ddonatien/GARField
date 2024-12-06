import igl
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.torch_version import TorchVersion

from models.position_encoding import get_embedder


class MeshSDF_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, vts, fcs):
        xyz_cpu = np.ascontiguousarray(xyz.detach().cpu())
        v_cpu = np.ascontiguousarray(vts.detach().cpu())
        f_cpu = np.ascontiguousarray(fcs.detach().cpu())
        igl_dists, igl_faces, _, igl_norms = igl.signed_distance(
            xyz_cpu, v_cpu, f_cpu, return_normals=True
        )

        wns = igl.fast_winding_number_for_meshes(v_cpu, f_cpu, xyz_cpu)
        wns[wns > 1] = 1
        igl_norms = (
            np.sign(igl_dists)[:, None] * (-2 * np.abs(wns) + 1)[:, None] * igl_norms
        )
        igl_dists = np.abs(igl_dists) * (-2 * np.abs(wns) + 1)

        norms = torch.tensor(igl_norms, dtype=torch.float32, requires_grad=False)
        faces = torch.tensor(igl_faces, dtype=torch.int64)
        ctx.mark_non_differentiable(faces)
        ctx.mark_non_differentiable(norms)
        fpts_id = fcs[faces]
        ctx.mark_non_differentiable(fpts_id)

        dists = torch.tensor(igl_dists, dtype=torch.float32).unsqueeze(-1)

        # Möller-Trumbore algorithm
        fpts = vts[fpts_id]
        T = xyz - fpts[:, 0]
        E1 = fpts[:, 1] - fpts[:, 0]
        E2 = fpts[:, 2] - fpts[:, 0]
        s = torch.det(torch.stack((norms, E1, E2), dim=-1))
        u = torch.det(torch.stack((norms, T, E2), dim=-1)) / s
        v = torch.det(torch.stack((norms, E1, T), dim=-1)) / s

        w = torch.stack((-u - v + 1, u, v), dim=-1)
        w = torch.clamp(w, min=0, max=1)
        ctx.mark_non_differentiable(w)

        ctx.save_for_backward(norms, w, fpts_id)
        ctx.dim = vts.shape
        return dists, w, fpts_id

    @staticmethod
    def backward(ctx, grad_dist_out, grad_coords_out, grad_ptsid_out):
        norms, w, fpts_id = ctx.saved_tensors
        vert_grads = torch.zeros(
            grad_dist_out.shape[0],
            ctx.dim[0],
            ctx.dim[1],
            device=grad_dist_out.device,
            requires_grad=False,
        )
        pt_grad = norms * grad_dist_out
        pt_idx = fpts_id.unsqueeze(-1).repeat(1, 1, 3)
        vert_grads = vert_grads.scatter(
            1, pt_idx, -w.unsqueeze(-1) * pt_grad.unsqueeze(1).repeat(1, 3, 1)
        )
        vg = vert_grads.sum(dim=0)
        return pt_grad, vg, None, None


class MeshSDF(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        n_layers,
        d_hidden,
        mesh_id=0,
        skip_in=tuple(),
        fc_s=8,
        multires=0,
        bias=0.5,
        scale=1.0,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,  # What is this ?
        residual=True,
        sdf_type=None,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers
        self.embed_fn_fine = None
        self.skip_in = skip_in
        self.scale = scale
        self.residual = residual
        self.mesh_id = mesh_id

        dims = [fc_s + 1] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.num_layers = len(dims)

        self.layers = nn.ModuleList()
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                if torch.__version__ > TorchVersion("1.13.0+cu117"):
                    lin = nn.utils.parametrizations.weight_norm(lin)
                else:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input_xyz, data_dict, details=None):
        v, f, p_e = data_dict["meshes"][self.mesh_id]

        dists, w, fpts_id = MeshSDF_func.apply(
            self.scale * input_xyz, self.scale * v, f
        )

        codes = p_e[fpts_id]

        codes = w.unsqueeze(-1) * codes
        codes = torch.sum(codes, dim=1)

        x = torch.cat((dists, codes), dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, dists, codes], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        if self.residual:
            out = (
                torch.cat(
                    [dists + x[..., :1] / self.scale, x[..., 1:]], dim=-1
                ).unsqueeze(0),
                (dists / self.scale).detach(),
            )
        else:
            out = (
                torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1).unsqueeze(0),
                (dists / self.scale).detach(),
            )
        return out

    def sdf(self, x, data_dict):
        return self.forward(x, data_dict)[0][..., :1]

    def gradient(self, x, data_dict):
        x.requires_grad_(True)
        y = self.sdf(x, data_dict)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]


class RigidSDF(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
        sdf_type=None,
    ):
        super().__init__()

        self.embed_fn_fine = None
        self.skip_in = skip_in
        self.scale = scale

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        # Input size: 39

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, -bias)
                    else:
                        nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                if torch.__version__ > TorchVersion("1.13.0+cu117"):
                    lin = nn.utils.parametrizations.weight_norm(lin)
                else:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input_xyz, data_dict, details=None):
        input_xyz = input_xyz * self.scale

        if self.embed_fn_fine is not None:
            input_xyz = self.embed_fn_fine(input_xyz)

        x = input_xyz
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_xyz], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1).unsqueeze(0), None

    def sdf(self, x, data_dict):
        return self.forward(x, data_dict)[0][:, :1]

    def gradient(self, x, data_dict):
        x.requires_grad_(True)
        y = self.sdf(x, data_dict)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)


class CompositeSDFNetwork(nn.Module):
    def __init__(self, init_kwargs):
        super().__init__()
        sdfs = []
        for sdf in init_kwargs:
            if init_kwargs[sdf]["sdf_type"] == "Mesh":
                model = MeshSDF(**init_kwargs[sdf])
            elif init_kwargs[sdf]["sdf_type"] == "Rigid":
                model = RigidSDF(**init_kwargs[sdf])
            else:
                continue
            sdfs.append(model)
        self.sdfs = nn.ModuleList(sdfs)

    def aggregate(self, values):
        dists = torch.abs(values[..., :1])
        mask = torch.zeros_like(dists)
        indexer = dists.argmin(0, True)
        out = torch.gather(
            values, 0, indexer.expand(*(-1,) * (dists.ndim - 1), values.shape[-1])
        )
        return out.squeeze(0), mask.scatter(0, indexer, 1.0).permute(2, 1, 0).squeeze(0)

    def forward(self, input_xyz, data_dict, details=None):
        a = []
        gts = []
        for m in self.sdfs:
            sdf, gt = m(input_xyz.clone(), data_dict, details)
            a.append(sdf)
            gts.append(gt)
        a = torch.cat(a, dim=0)
        if details is not None and "partial_sdf" in details:
            details["partial_sdf"] = a[..., :1]
        if details is not None and "dists_gt" in details:
            details["dists_gt"] = gts
        out, attn = self.aggregate(a)
        return out, attn

    def sdf(self, x, data_dict, details=None):
        return self.forward(x, data_dict, details)[0][:, :1]

    def gradient(self, x, data_dict):
        x.requires_grad_(True)
        y = self.sdf(x, data_dict)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)


# This implementation is partially borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires_view=0,
        squeeze_out=True,
        dropout=0.2,
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0 and self.mode == "idr":
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3
            dims[0] += 3

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                if torch.__version__ > TorchVersion("1.13.0+cu117"):
                    lin = nn.utils.parametrizations.weight_norm(lin)
                else:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_xyz, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == "idr":
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = self.dropout(rendering_input)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        d_in=3,
        d_in_view=3,
        multires=0,
        multires_view=0,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(
                multires_view, input_dims=d_in_view
            )
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [
                (
                    nn.Linear(W, W)
                    if i not in self.skips
                    else nn.Linear(W + self.input_ch, W)
                )
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        x = input_pts
        for i, l in enumerate(self.pts_linears):
            x = l(x)
            x = F.relu(x)
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(x)
            feature = self.feature_linear(x)
            x = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                x = self.views_linears[i](x)
                x = F.relu(x)

            rgb = self.rgb_linear(x)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
