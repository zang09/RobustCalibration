import torch

class SE3:
    def __init__(self, R=None, t=None, mat=None, se3=None) -> None:
        if mat is not None:
            if mat.shape[-2] == 4:
                self._mat = mat
            else:
                zeros = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
                if mat.dim() == 3:
                    self._mat = torch.cat((mat,zeros[None].expand(mat.shape[0],-1, -1)), dim=1)
                else:
                    self._mat = torch.row_stack((mat, zeros))
        elif R is not None:  
            if t is None:
                self._mat = R
            else:  
                self._mat = torch.eye(4, 4, dtype=R.dtype, device=R.device)
                if R.dim() == 3:
                    self._mat = self._mat[None].expand(R.shape[0], 4, 4)    
                self._mat[..., :3, :3] = R
                self._mat[..., :3, 3] = t 
        elif se3 is not None:
            Rt = Lie.se3_to_SE3(se3)               # (..., 3, 4)
            zeros_row = torch.zeros(Rt.shape[:-2] + (1, 4), dtype=Rt.dtype, device=Rt.device)
            zeros_row[..., 0, 3] = 1.0
            self._mat = torch.cat([Rt, zeros_row], dim=-2)  # (..., 4, 4)
        else:
            raise ValueError("Provide one of (mat, (R,t), se3).")
        self.mat = self._mat
        self.R = self._mat[..., :3, :3]
        self.t = None if (R is not None and t is None) else self._mat[..., :3, 3]

    def __matmul__(self, other):
 
        if isinstance(other, SE3):
            return SE3(mat=self._mat @ other._mat)
        elif isinstance(other, torch.Tensor):
            R, t = self.R, self.t
            if self._mat.dim()==2:
                if t is not None:
                    return (other @ R.transpose(0, 1) + t[None, :])
                else:
                    return other @ R.transpose(0, 1)
            else:
                if t is not None:
                    return torch.bmm(R, other[..., None])[..., 0] + t
                else:
                    return torch.bmm(R, other[..., None])[..., 0]
    
    def __getitem__(self, index):
        if self._mat.dim()<3:
            return self._mat[index]
        return SE3(mat=self._mat[index])
    
    def invert(self):
        inv = torch.linalg.inv(self._mat)
        if self.t is None:
            return SE3(R=inv)
        else:
            return SE3(mat=inv)
    
    @property
    def get_Rt(self):
        return SE3(R=self.R), self.t

    @property
    def get_R(self):
        return SE3(R=self.R)
    
    @property
    def get_t(self):
        return self.t
    
    def se3(self):
        return Lie.SE3_to_se3(self._mat[..., :3, :])

class Camera:
    @staticmethod
    def pixel_to_ray(width, height, intr, mask):
        v, u = torch.meshgrid(torch.arange(height, dtype=torch.float).cuda(), torch.arange(width, dtype=torch.float).cuda())
        uv_ = torch.stack((u, v, torch.ones_like(u)), dim=-1)[mask]
        xyz = intr.invert() @ uv_
        return xyz


class Lie:
    @staticmethod
    def so3_to_SO3(w):  # [..., 3]
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=w.dtype)
        I = I.expand(w.shape[:-1] + (3, 3))
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        R = I + A * wx + B * (wx @ wx)
        return R

    @staticmethod
    def SO3_to_so3(R, eps=1e-6):  # [..., 3, 3]  -- π-safe
        R = Lie._project_to_SO3(R)
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = (trace - 1.0) * 0.5
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.arccos(cos_theta)[..., None, None]

        K = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim=-1)

        w = torch.zeros(R.shape[:-2] + (3,), dtype=R.dtype, device=R.device)

        # General case (eps < theta < pi-eps)
        gen_mask = ((theta > eps) & (torch.pi - theta > eps))[..., 0, 0]
        if gen_mask.any():
            sin_theta = torch.sin(theta[gen_mask])
            w[gen_mask] = (theta[gen_mask][..., 0, 0] / (2.0 * sin_theta[..., 0, 0]))[..., None] * K[gen_mask]

        # Small angle approximation (theta ~ 0)
        small_mask = (theta <= eps)[..., 0, 0]
        if small_mask.any():
            w[small_mask] = 0.5 * K[small_mask]

        # Special handling near π
        pi_mask = (torch.pi - theta <= 1e-4)[..., 0, 0]
        if pi_mask.any():
            Rp = R[pi_mask]
            rx = torch.sqrt(torch.clamp((Rp[..., 0, 0] + 1) * 0.5, min=0))
            ry = torch.sqrt(torch.clamp((Rp[..., 1, 1] + 1) * 0.5, min=0))
            rz = torch.sqrt(torch.clamp((Rp[..., 2, 2] + 1) * 0.5, min=0))
            axis = torch.zeros(Rp.shape[:-2] + (3,), dtype=R.dtype, device=R.device)

            idx = torch.argmax(torch.stack([rx, ry, rz], dim=-1), dim=-1)

            # x max
            mask_x = idx == 0
            if mask_x.any():
                ax = rx[mask_x]
                axis_x = torch.stack([
                    ax,
                    (Rp[mask_x][..., 0, 1] + Rp[mask_x][..., 1, 0]) / (4 * torch.clamp(ax, min=eps)),
                    (Rp[mask_x][..., 0, 2] + Rp[mask_x][..., 2, 0]) / (4 * torch.clamp(ax, min=eps)),
                ], dim=-1)
                axis[mask_x] = axis_x
            # y max
            mask_y = idx == 1
            if mask_y.any():
                ay = ry[mask_y]
                axis_y = torch.stack([
                    (Rp[mask_y][..., 0, 1] + Rp[mask_y][..., 1, 0]) / (4 * torch.clamp(ay, min=eps)),
                    ay,
                    (Rp[mask_y][..., 1, 2] + Rp[mask_y][..., 2, 1]) / (4 * torch.clamp(ay, min=eps)),
                ], dim=-1)
                axis[mask_y] = axis_y
            # z max
            mask_z = idx == 2
            if mask_z.any():
                az = rz[mask_z]
                axis_z = torch.stack([
                    (Rp[mask_z][..., 0, 2] + Rp[mask_z][..., 2, 0]) / (4 * torch.clamp(az, min=eps)),
                    (Rp[mask_z][..., 1, 2] + Rp[mask_z][..., 2, 1]) / (4 * torch.clamp(az, min=eps)),
                    az,
                ], dim=-1)
                axis[mask_z] = axis_z

            axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=eps)
            w[pi_mask] = torch.pi * axis

        return w

    @staticmethod
    def se3_to_SE3(wu):  # [..., 6] -> [..., 3, 4]
        w, u = wu.split([3, 3], dim=-1)
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)
        C = Lie.taylor_C(theta)
        R = I + A * wx + B * (wx @ wx)
        V = I + B * wx + C * (wx @ wx)
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    @staticmethod
    def SE3_to_se3(Rt, eps=1e-8):  # [..., 3, 4] -> [..., 6]  -- θ≈π 안전화
        R, t = Rt.split([3, 1], dim=-1)  # R: (...,3,3), t: (...,3,1)
        R = Lie._project_to_SO3(R)
        w = Lie.SO3_to_so3(R)            # π-safe
        wx = Lie.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=Rt.device, dtype=Rt.dtype).expand(R.shape[:-2] + (3, 3))

        A = Lie.taylor_A(theta)
        B = Lie.taylor_B(theta)

        # V = I + B*wx + ((1 - A)/theta^2) * wx@wx  (Numerically unstable near small angles/π → safe inverse)
        theta2 = torch.clamp(theta**2, min=eps)
        V = I + B * wx + ((1 - A) / theta2) * (wx @ wx)

        # Add a small regularization for numerical stabilization before computing the inverse
        reg = 1e-8
        Vinv = torch.linalg.inv(V + reg * I)
        u = (Vinv @ t)[..., 0]
        return torch.cat([w, u], dim=-1)
    
    @staticmethod
    def _project_to_SO3(R):
        # R: (..., 3, 3)
        U, _, Vh = torch.linalg.svd(R)
        Rp = U @ Vh
        det = torch.det(Rp)
        if (det < 0).any():
            # Flip the last axis to remove reflection
            Vh_adj = Vh.clone()
            Vh_adj[det < 0, -1, :] *= -1
            Rp[det < 0] = U[det < 0] @ Vh_adj[det < 0]
        return Rp

    @staticmethod
    def skew_symmetric(w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    @staticmethod
    def taylor_A(x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    @staticmethod
    def taylor_B(x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
    @staticmethod
    def taylor_C(x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

class Quaternion:
    """
    MIT License

    Copyright (c) 2021 Chen-Hsuan Lin

    Reference https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/LICENSE
    """
    @staticmethod
    def to_vector(r):
        w, x, y, z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        norm = 2 * w / (w * w + y * y + z * z)
        v = torch.column_stack((norm * w - 1, norm * z, -norm * y))
        return v

    # from gaussian splatting    
    @staticmethod
    def to_mat(r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return R