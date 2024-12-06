import os
import igl
import torch
import trimesh
import logging

import numpy as np
import torch.nn as nn

from glob import glob
from .LieAlgebra import se3
from torch_geometric.data import Data
from models.position_encoding import LapEncoding


def get_pe(mesh, norm, use_edge_attr, size):
    edge_index = igl.edges(mesh.faces)
    edge_tensor = torch.tensor(edge_index).T.contiguous()
    # Make edges bi-directional
    edge_index = torch.cat((edge_tensor,
                            torch.roll(edge_tensor, 1, dims=0)), dim=1)

    pos_np = mesh.vertices
    pos = torch.tensor(pos_np)
    edge_attr = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=-1).unsqueeze(-1)

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    enc = LapEncoding(size, use_edge_attr=use_edge_attr, normalization=norm)
    lpe = enc.compute_pe(data)

    return lpe


class MobileStlContainer(nn.Module):
    def __init__(self, mesh_file, scale_mat, transform_mats, rigid_adjust=False, dropout=0.2,
                 lpe_norm='sym', lpe_use_edge_attr=True, lpe_size=16, code_bias=False, code_learnt=False, use_lpe=True):
        super().__init__()
        logging.getLogger("trimesh").setLevel(logging.ERROR)
        self.rigid_adjust = rigid_adjust
        self.dropout = nn.Dropout(p=dropout)

        print(f"Loading mesh {mesh_file}")
        stl_mesh = trimesh.load(mesh_file, process=True, maintain_order=True)
        pos_enc = get_pe(stl_mesh, lpe_norm, lpe_use_edge_attr, lpe_size)
        self.code_bias = code_bias
        self.use_lpe = use_lpe
        if self.code_bias:
            print("Code bias")
            s = 1
            if code_learnt and self.use_lpe:
                s = 1e-4
            self.obj_code = nn.Parameter(data=s * torch.rand(stl_mesh.vertices.shape[0], lpe_size),
                                         requires_grad=code_learnt)
        self.faces = nn.Parameter(data=torch.tensor(stl_mesh.faces), requires_grad=False)
        self.verts = nn.Parameter(data=torch.tensor(stl_mesh.vertices).float(), requires_grad=False)
        self.pos_enc = nn.Parameter(data=pos_enc, requires_grad=False)
        print(f"Mesh {mesh_file} loaded !")

        # T = trimesh.transformations.translation_matrix(np.array([0, -0.009, 0.002]))
        base_transforms = (transform_mats.transpose(1, 2) @ torch.inverse(scale_mat).T).transpose(1, 2)
        transforms = []
        for tmat in base_transforms.cpu().numpy():
            # transforms.append(torch.tensor(tmat@T).float())
            transforms.append(torch.tensor(tmat).float())
        self.transforms = nn.Parameter(data=torch.stack(transforms, dim=0), requires_grad=False)
        if self.rigid_adjust:
            self.adjust_transform = nn.Parameter(data=torch.zeros(6), requires_grad=True)

    def get_pos_codes(self):
        if self.code_bias:
            out = self.pos_enc.data.clone() + self.obj_code
        else:
            out = self.pos_enc.data.clone()
        return out

    def forward(self, idx):
        v = self.verts
        f = self.faces

        count = v.shape[0]
        hv = torch.cat([v, torch.ones((count, 1))], dim=-1)
        if self.rigid_adjust:
            trs = self.transforms[idx] @ se3.exp(self.adjust_transform)
        else:
            trs = self.transforms[idx]
        v = (trs @ hv.T).T[:, :3]
        if self.use_lpe:
            p_e = self.pos_enc.to(v)
            if self.code_bias:
                p_e = p_e + self.obj_code
        else:
            p_e = self.obj_code
        return v, f, self.dropout(p_e)


class ListStlContainer(nn.Module):
    def __init__(self, mesh_pattern, scale_mat=None, transform_mats=None, rigid_adjust=False, dropout=0.2,
                 lpe_norm='sym', lpe_use_edge_attr=True, lpe_size=16, code_bias=False):
        super().__init__()
        logging.getLogger("trimesh").setLevel(logging.ERROR)
        self.rigid_adjust = rigid_adjust
        self.dropout = nn.Dropout(p=dropout)

        print(f"Loading meshes {mesh_pattern}")
        faces = None
        verts = []
        mesheslis = sorted(glob(mesh_pattern))
        for i, mesh_name in enumerate(mesheslis):
            mesh = trimesh.load(mesh_name, process=True, maintain_order=True)
            if len(verts) > 0:
                assert mesh.vertices.shape[0] == verts[-1].shape[0]
            if faces is not None:
                assert np.sum((mesh.faces - faces)**2) == 0
            if i == 0:
                pos_enc = get_pe(mesh, lpe_norm, lpe_use_edge_attr, lpe_size)
            verts.append(torch.tensor(mesh.vertices).float())
            faces = mesh.faces
        self.code_bias = code_bias
        if self.code_bias:
            self.obj_code = nn.Parameter(data=torch.rand(lpe_size), requires_grad=False)
        self.faces = nn.Parameter(data=torch.tensor(faces), requires_grad=False)
        self.verts = nn.Parameter(data=torch.stack(verts, dim=0), requires_grad=False)
        self.pos_enc = nn.Parameter(data=pos_enc, requires_grad=False)
        print(f"Meshes {mesh_pattern} loaded !")

        if scale_mat is not None and transform_mats is not None :
            base_transforms = (transform_mats.transpose(1, 2) @ torch.inverse(scale_mat).T).transpose(1, 2)
            transforms = []
            for tmat in base_transforms.cpu().numpy():
                transforms.append(torch.tensor(tmat@T).float())
            self.transforms = nn.Parameter(data=torch.stack(transforms, dim=0), requires_grad=False)
        else:
            self.transforms = None
        if self.rigid_adjust:
            self.adjust_transform = nn.Parameter(data=torch.zeros(6), requires_grad=True)

    def get_pos_codes(self):
        if self.code_bias:
            out = self.pos_enc.data.clone() + self.obj_code
        else:
            out = self.pos_enc.data.clone()
        return out

    def forward(self, idx):
        v = self.verts[idx]
        f = self.faces
        p_e = self.pos_enc.to(v)
        if self.transforms is not None:
            count = v.shape[0]
            hv = torch.cat([v, torch.ones((count, 1))], dim=-1)
            if self.rigid_adjust:
                trs = self.transforms[idx] @ se3.exp(self.adjust_transform)
            else:
                trs = self.transforms[idx]
            v = (trs @ hv.T).T[:, :3]
        if self.code_bias:
            p_e = p_e + self.obj_code
        # p_e = self.obj_code
        return v, f, self.dropout(p_e)
