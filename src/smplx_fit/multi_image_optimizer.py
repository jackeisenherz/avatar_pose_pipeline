
# Drop-in replacement for: src/smplx_fit/multi_image_optimizer.py

import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from .renderer import SilhouetteRenderer
from .utils import load_pose_json, load_visibility_json
from .joint_mapper import SMPLXJointMapper
from .body_regions import BodyRegionWeights
from .region_masks import RegionAwareMasks
from .losses import silhouette_loss, shape_prior_loss, pose_prior_loss, translation_loss


class MultiImageOptimizer:
    COCO_LIMBS=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    BODY_JOINTS=[5,6,7,8,9,10,11,12,13,14,15,16]
    BAND_SPECS={
        "chest":(0.22,0.43),
        "breast":(0.25,0.42),
        "underbust":(0.38,0.50),
        "waist":(0.40,0.58),
        "abdomen":(0.44,0.64),
        "hips":(0.54,0.78),
        "glutes":(0.58,0.82)
    }

    def __init__(
        self,
        model_path,
        gender="female",
        image_size=512,
        pseudo_weight=0.0,
        optimize_focal=True,
        base_focal=1500.0,
        debug=True,
        debug_every=25,
        debug_max_images=6,
        use_pose_quality_metadata=True,
        pose_quality_full_threshold=0.65,
        pose_quality_low_threshold=0.50,
        pose_quality_very_low_threshold=0.35,
        pose_quality_skip_keypoints_threshold=0.25,
        pose_lock_weight=1.0,
        mask_stats_weight=1.0,
        debug_keypoints=True,
        use_hair_aware_masks=True,
        hair_unknown_weight=0.0,
        tight_fit_weight=1.0,
        outside_target_weight=1.0,
        core_coverage_weight=1.0,
        core_coverage_adaptive_max=1.75,
        arm_silhouette_reduction=1.0,
        pose_core_dilate_px=19,
        min_keypoint_conf=0.20,
        width_weight=1.25,
        area_weight=0.65,
        anti_bloat_weight=0.35,
        breast_preserve_weight=1.0,
        glute_preserve_weight=0.9,
        use_chest_offsets=True,
        chest_offset_limit=0.010,
        chest_offset_weight=0.55,
        chest_project_weight=1.25,
        chest_width_weight=1.50,
        chest_area_weight=1.00,
        chest_smooth_weight=300.0,
        chest_offset_l2_weight=40.0,
        chest_outward_prior_weight=0.40,
        chest_inward_prior_weight=0.05,
        chest_smooth_steps=10,
        chest_smooth_alpha=0.70,
        bilateral_breast_weight=0.55,
        bilateral_centroid_weight=0.25,
        abdomen_guard_weight=1.25,
        abdomen_area_weight=0.90,
        waist_guard_weight=0.80,
        glute_width_weight=0.90,
        glute_area_weight=0.85,
        glute_bloat_weight=0.40,
        lower_body_kp_weight=2.5,
        lower_body_center_weight=1.5,
        thigh_direction_weight=1.2,
        lower_body_reproj_weight=0.10,
        lower_body_pose_gate_threshold=0.45,
        pose_anchor_weight=250.0,
        joint_anchor_weight=120.0,
        camera_anchor_weight=60.0,
        beta_anchor_weight=8.0,
        silhouette_guard_weight=120.0,
        anchor_sil_tolerance=0.008,
        anchor_outside_tolerance=0.003,
        anchor_iou_tolerance=0.008,
        chest_residual_weight=3.0,
        chest_outside_guard_weight=6.0,
        anchor_reproj_threshold=15.0,
        anchor_sil_threshold=0.42,
        anchor_outside_threshold=0.022,
        delay_refinement_until_anchor=True,
        per_image_focal_regularization_weight=0.10,
        normalize_lower_body_loss=True,
    ):
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.image_size=int(image_size)
        self.pseudo_weight=float(pseudo_weight)
        self.optimize_focal=bool(optimize_focal)
        self.base_focal=float(base_focal)
        self.debug=bool(debug)
        self.debug_every=int(debug_every)
        self.debug_max_images=int(debug_max_images)
        self.use_pose_quality_metadata=bool(use_pose_quality_metadata)
        self.pose_quality_full_threshold=float(pose_quality_full_threshold)
        self.pose_quality_low_threshold=float(pose_quality_low_threshold)
        self.pose_quality_very_low_threshold=float(pose_quality_very_low_threshold)
        self.pose_quality_skip_keypoints_threshold=float(pose_quality_skip_keypoints_threshold)
        self.pose_lock_weight=float(pose_lock_weight)
        self.mask_stats_weight=float(mask_stats_weight)
        self.debug_keypoints=bool(debug_keypoints)
        self.use_hair_aware_masks=bool(use_hair_aware_masks)
        self.hair_unknown_weight=float(hair_unknown_weight)
        self.tight_fit_weight=float(tight_fit_weight)
        self.outside_target_weight=float(outside_target_weight)
        self.core_coverage_weight=float(core_coverage_weight)
        self.core_coverage_adaptive_max=float(core_coverage_adaptive_max)
        self.arm_silhouette_reduction=float(arm_silhouette_reduction)
        self.pose_core_dilate_px=int(pose_core_dilate_px)
        self.min_keypoint_conf=float(min_keypoint_conf)

        self.width_weight=float(width_weight)
        self.area_weight=float(area_weight)
        self.anti_bloat_weight=float(anti_bloat_weight)
        self.breast_preserve_weight=float(breast_preserve_weight)
        self.glute_preserve_weight=float(glute_preserve_weight)

        self.use_chest_offsets=bool(use_chest_offsets)
        self.chest_offset_limit=float(chest_offset_limit)
        self.chest_offset_weight=float(chest_offset_weight)
        self.chest_project_weight=float(chest_project_weight)
        self.chest_width_weight=float(chest_width_weight)
        self.chest_area_weight=float(chest_area_weight)
        self.chest_smooth_weight=float(chest_smooth_weight)
        self.chest_offset_l2_weight=float(chest_offset_l2_weight)
        self.chest_outward_prior_weight=float(chest_outward_prior_weight)
        self.chest_inward_prior_weight=float(chest_inward_prior_weight)
        self.chest_smooth_steps=int(chest_smooth_steps)
        self.chest_smooth_alpha=float(chest_smooth_alpha)
        self.bilateral_breast_weight=float(bilateral_breast_weight)
        self.bilateral_centroid_weight=float(bilateral_centroid_weight)

        self.abdomen_guard_weight=float(abdomen_guard_weight)
        self.abdomen_area_weight=float(abdomen_area_weight)
        self.waist_guard_weight=float(waist_guard_weight)
        self.glute_width_weight=float(glute_width_weight)
        self.glute_area_weight=float(glute_area_weight)
        self.glute_bloat_weight=float(glute_bloat_weight)
        self.lower_body_kp_weight=float(lower_body_kp_weight)
        self.lower_body_center_weight=float(lower_body_center_weight)
        self.thigh_direction_weight=float(thigh_direction_weight)
        self.lower_body_reproj_weight=float(lower_body_reproj_weight)
        self.lower_body_pose_gate_threshold=float(lower_body_pose_gate_threshold)
        self.pose_anchor_weight=float(pose_anchor_weight)
        self.joint_anchor_weight=float(joint_anchor_weight)
        self.camera_anchor_weight=float(camera_anchor_weight)
        self.beta_anchor_weight=float(beta_anchor_weight)
        self.silhouette_guard_weight=float(silhouette_guard_weight)
        self.anchor_sil_tolerance=float(anchor_sil_tolerance)
        self.anchor_outside_tolerance=float(anchor_outside_tolerance)
        self.anchor_iou_tolerance=float(anchor_iou_tolerance)
        self.chest_residual_weight=float(chest_residual_weight)
        self.chest_outside_guard_weight=float(chest_outside_guard_weight)
        self.anchor_reproj_threshold=float(anchor_reproj_threshold)
        self.anchor_sil_threshold=float(anchor_sil_threshold)
        self.anchor_outside_threshold=float(anchor_outside_threshold)
        self.delay_refinement_until_anchor=bool(delay_refinement_until_anchor)
        self.per_image_focal_regularization_weight=float(per_image_focal_regularization_weight)
        self.normalize_lower_body_loss=bool(normalize_lower_body_loss)

        self.model=smplx.create(
            model_path=model_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=10,
            ext="npz",
        ).to(self.device)

        self.renderer=SilhouetteRenderer(image_size=image_size, device=self.device)
        self.region_maps=RegionAwareMasks.screen_region_maps(self.image_size, self.image_size, self.device)
        self.sil_region_weights=RegionAwareMasks.silhouette_region_weights(self.image_size, self.image_size, self.device)
        self.anti_bloat_map=RegionAwareMasks.anti_bloat_weights(self.image_size, self.image_size, self.device)

        yy=torch.linspace(0.0,1.0,self.image_size,device=self.device).view(1,1,self.image_size,1)
        xx=torch.linspace(0.0,1.0,self.image_size,device=self.device).view(1,1,1,self.image_size)
        self.grid_x=xx.expand(1,1,self.image_size,self.image_size)
        self.grid_y=yy.expand(1,1,self.image_size,self.image_size)

        self.joint_weights=torch.tensor(
            [0.20,0.10,0.10,0.10,0.10,2.50,2.50,3.00,3.00,3.50,3.50,2.50,2.50,2.50,2.50,2.00,2.00],
            dtype=torch.float32, device=self.device
        ).view(1,17)

        self._init_chest_region()

    def _init_chest_region(self):
        with torch.no_grad():
            out=self.model(
                betas=torch.zeros(1,10,device=self.device),
                body_pose=torch.zeros(1,63,device=self.device),
                global_orient=torch.zeros(1,3,device=self.device),
                transl=None,
                return_verts=True,
            )
        tv=out.vertices[0].detach()
        self.template_vertices=tv
        faces=self.model.faces_tensor.detach().long().to(self.device)
        cvm=self._select_chest_vertices(tv)
        cids=torch.where(cvm)[0]
        if cids.numel()<50:
            cvm=self._select_chest_vertices(tv, broad=True)
            cids=torch.where(cvm)[0]
        fm=cvm[faces[:,0]] & cvm[faces[:,1]] & cvm[faces[:,2]]
        cf=faces[fm]
        if cf.shape[0] < 50:
            fm=((cvm[faces[:,0]].long()+cvm[faces[:,1]].long()+cvm[faces[:,2]].long())>=2)
            cf=faces[fm]
        self.chest_vertex_mask=cvm
        self.chest_vertex_ids=cids
        self.chest_faces=cf.unsqueeze(0)
        self.chest_face_mask_np=fm.detach().cpu().numpy().astype(bool)
        self.template_normals=self._vertex_normals(tv.unsqueeze(0), faces).detach()
        self.chest_edges=self._build_chest_edges(faces, cvm)
        self.chest_vertex_weights=self._chest_vertex_weights(tv, cids).view(1,-1,1).detach()
        chest_sub=tv[cids]
        x_mid=chest_sub[:,0].median()
        self.chest_left_local_mask=(chest_sub[:,0] < x_mid).detach()
        self.chest_right_local_mask=(chest_sub[:,0] >= x_mid).detach()

        x=chest_sub[:,0]
        y=chest_sub[:,1]
        x_span=(x.max()-x.min()).clamp(min=1e-6)
        y_norm=(y-y.min())/(y.max()-y.min()).clamp(min=1e-6)
        self.chest_sternum_local_mask=(
            (torch.abs(x-x_mid)<0.13*x_span) &
            (y_norm>0.18) &
            (y_norm<0.78)
        ).detach()

    def _select_chest_vertices(self, v, broad=False):
        x,y,z=v[:,0],v[:,1],v[:,2]
        xn=(x-x.min())/(x.max()-x.min()).clamp(min=1e-8)
        yn=(y-y.min())/(y.max()-y.min()).clamp(min=1e-8)
        zn=(z-z.min())/(z.max()-z.min()).clamp(min=1e-8)
        if broad:
            return (yn>0.48)&(yn<0.76)&(xn>0.16)&(xn<0.84)&(zn>0.34)
        return (yn>0.53)&(yn<0.73)&(xn>0.20)&(xn<0.80)&(zn>0.42)

    def _chest_vertex_weights(self, v, cids):
        sub=v[cids]
        x,y,z=sub[:,0],sub[:,1],sub[:,2]
        xn=(x-x.min())/(x.max()-x.min()).clamp(min=1e-8)
        yn=(y-y.min())/(y.max()-y.min()).clamp(min=1e-8)
        zn=(z-z.min())/(z.max()-z.min()).clamp(min=1e-8)
        left=torch.exp(-(((xn-0.34)**2)/0.040+((yn-0.49)**2)/0.070))
        right=torch.exp(-(((xn-0.66)**2)/0.040+((yn-0.49)**2)/0.070))
        lobe=torch.maximum(left,right)
        frontal=torch.clamp((zn-0.35)/0.65,0.0,1.0)

        # Critical: do not let the whole center chest inflate like a single
        # continuous "top". The breast offset layer should primarily act on
        # two lateral lobes, while the sternum/inter-breast zone remains flat.
        center_gap=torch.exp(-(((xn-0.50)**2)/0.010+((yn-0.50)**2)/0.090))
        weights=(0.03+0.82*lobe+0.15*frontal)*(1.0-0.92*center_gap)
        return weights.clamp(0.01,1.00)

    def _build_chest_edges(self, faces, cvm):
        edges=torch.cat([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]], dim=0)
        em=cvm[edges[:,0]] & cvm[edges[:,1]]
        edges=torch.sort(edges[em], dim=1).values
        edges=torch.unique(edges, dim=0)
        g2l=torch.full((int(cvm.shape[0]),), -1, dtype=torch.long, device=self.device)
        cids=torch.where(cvm)[0]
        g2l[cids]=torch.arange(cids.numel(), device=self.device)
        le=g2l[edges]
        return le[(le[:,0]>=0)&(le[:,1]>=0)].long()

    def _vertex_normals(self, vertices, faces):
        v=vertices[0]; f=faces.long()
        v0,v1,v2=v[f[:,0]],v[f[:,1]],v[f[:,2]]
        fn=torch.cross(v1-v0, v2-v0, dim=-1)
        n=torch.zeros_like(v)
        n.index_add_(0, f[:,0], fn)
        n.index_add_(0, f[:,1], fn)
        n.index_add_(0, f[:,2], fn)
        n=F.normalize(n, dim=-1, eps=1e-8)
        return n.unsqueeze(0)

    def _smooth_chest_scalars(self, scalars, steps=None, alpha=None):
        steps=self.chest_smooth_steps if steps is None else steps
        alpha=self.chest_smooth_alpha if alpha is None else alpha
        if steps<=0 or self.chest_edges.numel()==0:
            return scalars
        out=scalars
        src=self.chest_edges[:,0]
        dst=self.chest_edges[:,1]
        for _ in range(steps):
            base=out[0]
            sums=torch.zeros_like(base)
            counts=torch.zeros(base.shape[0],1,dtype=base.dtype,device=base.device)
            sums.index_add_(0, src, base[dst]); sums.index_add_(0, dst, base[src])
            ones=torch.ones(src.shape[0],1,dtype=base.dtype,device=base.device)
            counts.index_add_(0, src, ones); counts.index_add_(0, dst, ones)
            nm=sums/counts.clamp(min=1.0)
            out=((1.0-alpha)*base + alpha*nm).unsqueeze(0)
        return out

    def _chest_offsets_full(self, cos):
        full=torch.zeros(1, self.template_normals.shape[1], 3, dtype=torch.float32, device=self.device)
        if cos is None or not self.use_chest_offsets:
            return full
        scalars=torch.clamp(cos, -self.chest_offset_limit, self.chest_offset_limit)
        scalars=self._smooth_chest_scalars(scalars)
        normals=self.template_normals[:, self.chest_vertex_ids, :]
        local_weights=self.chest_vertex_weights.clone()

        # Keep sternum / inter-breast vertices nearly fixed so the optimizer
        # cannot create a tent-like bridge between the breasts through the
        # local chest offset layer.
        if hasattr(self, "chest_sternum_local_mask"):
            local_weights[:, self.chest_sternum_local_mask, :] *= 0.05

        offsets=normals * scalars * local_weights
        full[:, self.chest_vertex_ids, :]=offsets
        return full

    def _chest_regularization(self, cos):
        if cos is None or not self.use_chest_offsets:
            z=torch.tensor(0.0, device=self.device)
            return z,z,z,z
        s=torch.clamp(cos, -self.chest_offset_limit, self.chest_offset_limit)
        ss=self._smooth_chest_scalars(s)
        l2=(ss**2).mean()
        if self.chest_edges.numel()>0:
            src,dst=self.chest_edges[:,0], self.chest_edges[:,1]
            sm=((ss[:,src]-ss[:,dst])**2).mean()
        else:
            sm=torch.tensor(0.0, device=self.device)
        out=(torch.relu(ss)**2).mean()
        inward=(torch.relu(-ss)**2).mean()
        return l2,sm,out,inward

    def _make_camera(self, f, cc, t):
        bs=t.shape[0]
        R=torch.eye(3, device=self.device).unsqueeze(0).repeat(bs,1,1)
        isz=torch.tensor([[self.image_size,self.image_size]], dtype=torch.float32, device=self.device).repeat(bs,1)
        if f.shape[0]==1 and bs>1: f=f.repeat(bs,1)
        if cc.shape[0]==1 and bs>1: cc=cc.repeat(bs,1)
        return PerspectiveCameras(focal_length=f, principal_point=cc, R=R, T=t, image_size=isz, in_ndc=False, device=self.device)

    def _project_points_screen(self, p, f, cc, t):
        cam=self._make_camera(f, cc, t)
        s=cam.transform_points_screen(p, image_size=((self.image_size,self.image_size),), with_xyflip=True)
        return s[:,:,:2]

    def _current_focal(self, lfs, image_index=None):
        """
        Return focal length for either one image or all images.

        Previous versions used one shared focal for all images. This version uses
        per-image focal log-scales so different crops / zoom levels / camera
        distances do not have to be absorbed by body shape or translation.
        """
        if lfs is None:
            if image_index is None:
                fv=torch.tensor([self.base_focal], dtype=torch.float32, device=self.device)
            else:
                fv=torch.tensor([self.base_focal], dtype=torch.float32, device=self.device)
        else:
            if image_index is None:
                fv=self.base_focal*torch.exp(lfs)
            else:
                fv=self.base_focal*torch.exp(lfs[image_index:image_index+1])
        fv=torch.clamp(fv, 800.0, 4000.0)
        return torch.stack([fv,fv], dim=-1).view(-1,2)

    def _focal_regularization(self, lfs):
        """
        Per-image focal regularization: keep individual focal values near their
        shared mean while still allowing crop/zoom differences.
        """
        if lfs is None:
            return torch.tensor(0.0, device=self.device)
        center=lfs.mean().detach()
        return (lfs**2).mean() + self.per_image_focal_regularization_weight*((lfs-center)**2).mean()

    def _load_image_shape(self, ip):
        img=cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if img is None: raise RuntimeError(f"Could not load image: {ip}")
        return img.shape[:2]


    def _load_pose_json_full(self, pose_json_path):
        """
        Load the full pose JSON so the optimizer can use the additional
        metadata produced by the improved pose estimator.

        The older load_pose_json helper only returns keypoints. This method
        preserves fields such as pose_quality_score, low_quality_pose,
        very_low_quality_pose, keypoint_alpha_support and joint_sanity.
        """
        with open(pose_json_path, "r") as f:
            data=json.load(f)
        return data

    def _pose_quality_info(self, pose_data):
        """
        Convert optional pose metadata into optimizer weights.

        Returns:
            dict with:
              quality:           0..1 quality score
              image_weight:      mild image-level downweight
              keypoint_weight:   direct kp/bone/center downweight
              is_low:            low-quality pose flag
              is_very_low:       very-low-quality pose flag
        """
        if not self.use_pose_quality_metadata:
            return {
                "quality":1.0,
                "image_weight":1.0,
                "keypoint_weight":1.0,
                "is_low":False,
                "is_very_low":False,
                "support":1.0,
                "sanity":1.0,
                "backend":"unknown",
                "model":"unknown",
            }

        # Prefer explicit quality score from the new pose estimator.
        quality=pose_data.get("pose_quality_score", None)

        if quality is None:
            # Fallback for old YOLO-only JSONs.
            try:
                confs=[float(k.get("confidence", 0.0)) for k in pose_data.get("keypoints", [])]
                quality=float(np.mean(confs)) if len(confs)>0 else 1.0
            except Exception:
                quality=1.0

        quality=float(np.clip(quality, 0.0, 1.0))

        support=float(pose_data.get("keypoint_alpha_support", 1.0))
        support=float(np.clip(support, 0.0, 1.0))

        sanity_data=pose_data.get("joint_sanity", {})
        if isinstance(sanity_data, dict):
            sanity=float(sanity_data.get("sanity_score", 1.0))
        else:
            sanity=1.0
        sanity=float(np.clip(sanity, 0.0, 1.0))

        is_low=bool(pose_data.get("low_quality_pose", quality < self.pose_quality_low_threshold))
        is_very_low=bool(pose_data.get("very_low_quality_pose", quality < self.pose_quality_very_low_threshold))

        # Keypoint terms should be strongly reduced when the pose estimate is
        # untrusted, because bad 2D joints can pull the whole SMPL-X fit away
        # from the silhouette.
        if quality < self.pose_quality_skip_keypoints_threshold:
            keypoint_weight=0.05
        elif is_very_low or quality < self.pose_quality_very_low_threshold:
            keypoint_weight=0.15
        elif is_low or quality < self.pose_quality_low_threshold:
            keypoint_weight=0.40
        elif quality < self.pose_quality_full_threshold:
            keypoint_weight=0.75
        else:
            keypoint_weight=1.0

        # Image-level weight is milder: even with weak pose, the silhouette,
        # region masks, and alpha are still useful.
        if is_very_low or quality < self.pose_quality_very_low_threshold:
            image_weight=0.55
        elif is_low or quality < self.pose_quality_low_threshold:
            image_weight=0.70
        elif quality < self.pose_quality_full_threshold:
            image_weight=0.85
        else:
            image_weight=1.0

        # If the keypoints are mostly outside the alpha mask, reduce trust
        # further. This catches wrong-person detections and wall-poster people.
        if support < 0.35:
            keypoint_weight*=0.25
            image_weight*=0.75
        elif support < 0.55:
            keypoint_weight*=0.50
            image_weight*=0.90

        # Sanity failures reduce pose loss, but less aggressively than alpha
        # support because unusual poses can legitimately look odd.
        if sanity < 0.35:
            keypoint_weight*=0.50
        elif sanity < 0.55:
            keypoint_weight*=0.75

        return {
            "quality":float(quality),
            "image_weight":float(np.clip(image_weight, 0.20, 1.25)),
            "keypoint_weight":float(np.clip(keypoint_weight, 0.00, 1.25)),
            "is_low":bool(is_low),
            "is_very_low":bool(is_very_low),
            "support":float(support),
            "sanity":float(sanity),
            "backend":str(pose_data.get("pose_backend", "unknown")),
            "model":str(pose_data.get("pose_model", "unknown")),
        }

    def _load_scaled_pose(self, pj, ip):
        pd=load_pose_json(pj)
        kp=pd["keypoints"].astype(np.float32)
        ih,iw=self._load_image_shape(ip)
        kp[:,0]*=self.image_size/float(iw)
        kp[:,1]*=self.image_size/float(ih)
        # Some pose backends produce scores slightly above 1.0.
        # Clamp to avoid overweighting individual joints.
        kp[:,2]=np.clip(kp[:,2], 0.0, 1.0)
        return kp

    def _find_background_debug_mask(self, image_path, suffix):
        """
        Find masks produced by the hair-aware background remover.

        Expected examples:
            data/output/03_normalized/img.png
            data/output/02_no_background/_debug_masks/img_body_fit_mask.png
            data/output/02_no_background/_debug_masks/img_silhouette_validity.png

        Also works when image_path itself already lives in 02_no_background.
        """
        image_path=Path(image_path)
        candidates=[]

        # Most common pipeline layout: 03_normalized -> sibling 02_no_background.
        candidates.append(image_path.parent.parent / "02_no_background" / "_debug_masks")

        # If the optimizer is run directly on 02_no_background images.
        candidates.append(image_path.parent / "_debug_masks")

        # If output folder names differ but the debug dir is a sibling.
        candidates.append(image_path.parent.parent / "_debug_masks")

        names=[
            f"{image_path.stem}_{suffix}.png",
            f"{image_path.stem}_{suffix}(1).png",
        ]

        for debug_dir in candidates:
            for name in names:
                p=debug_dir / name
                if p.exists():
                    return p

        return None

    def _read_mask_file(self, path, threshold=10):
        img=cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim==3 and img.shape[2]==4:
            img=img[:,:,3]
        elif img.ndim==3:
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img=cv2.resize(
            img,
            (self.image_size,self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        return (img>threshold).astype(np.float32)

    def _load_target_mask(self, ip):
        """
        Load silhouette target.

        If the hair-aware background remover produced a body_fit_mask, use it.
        Otherwise fall back to the alpha channel of the normalized image.

        This prevents long loose hair from being treated as body volume.
        """
        used_hair_mask=False

        if self.use_hair_aware_masks:
            body_fit_path=self._find_background_debug_mask(ip, "body_fit_mask")
            if body_fit_path is not None:
                mask=self._read_mask_file(body_fit_path, threshold=10)
                if mask is not None:
                    used_hair_mask=True
                    return (
                        torch.tensor(mask, dtype=torch.float32, device=self.device)
                        .unsqueeze(0).unsqueeze(0)
                    )

        rgba=cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"Could not load image: {ip}")
        if rgba.ndim==3 and rgba.shape[2]==4:
            alpha=rgba[:,:,3]
        else:
            gray=cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha=(gray>5).astype(np.uint8)*255
        mask=(alpha>10).astype(np.float32)
        mask=cv2.resize(mask, (self.image_size,self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def _load_silhouette_validity(self, ip):
        """
        Load per-pixel silhouette validity.

        Hair-covered pixels should be unknown, not background:
            target_body_mask = 0
            validity = 0

        The validity mask multiplies silhouette/distance/IoU/edge/stat losses.
        If no validity mask exists, all pixels are valid.
        """
        if self.use_hair_aware_masks:
            validity_path=self._find_background_debug_mask(ip, "silhouette_validity")
            if validity_path is not None:
                valid=self._read_mask_file(validity_path, threshold=10)
                if valid is not None:
                    return (
                        torch.tensor(valid, dtype=torch.float32, device=self.device)
                        .unsqueeze(0).unsqueeze(0)
                    )

            # Fallback: if a hair_remove mask exists but no explicit validity
            # file exists, treat those hair pixels as unknown.
            hair_remove_path=self._find_background_debug_mask(ip, "hair_remove")
            if hair_remove_path is not None:
                hair=self._read_mask_file(hair_remove_path, threshold=10)
                if hair is not None:
                    valid=1.0-hair
                    if self.hair_unknown_weight>0.0:
                        valid=np.maximum(valid, self.hair_unknown_weight).astype(np.float32)
                    return (
                        torch.tensor(valid, dtype=torch.float32, device=self.device)
                        .unsqueeze(0).unsqueeze(0)
                    )

        return torch.ones(
            1,1,self.image_size,self.image_size,
            dtype=torch.float32,
            device=self.device,
        )

    def _validity_stats(self, validity):
        return float(validity.mean().detach().cpu().item())

    def _bbox_from_mask(self, m):
        ys,xs=np.where(m>0.5)
        if len(xs)==0: return None
        return {"cx":float(xs.mean()), "cy":float(ys.mean()), "w":int(xs.max()-xs.min()+1), "h":int(ys.max()-ys.min()+1), "area":float(m.mean())}

    def _initial_translation_from_mask_and_keypoints(self, m, kp):
        bb=self._bbox_from_mask(m)
        if bb is None:
            return torch.tensor([[0.0,0.0,5.0]], dtype=torch.float32, device=self.device)
        reliable=kp[:,2]>self.min_keypoint_conf
        reliable_body=reliable[self.BODY_JOINTS]
        if reliable_body.sum()>=4:
            bk=kp[self.BODY_JOINTS][reliable_body]
            cx,cy=float(bk[:,0].mean()), float(bk[:,1].mean())
        else:
            cx,cy=float(bb["cx"]), float(bb["cy"])
        h_px=max(float(bb["h"]), 32.0)
        z=float(np.clip(self.base_focal*1.65/h_px, 2.5, 8.0))
        tx=(cx-self.image_size/2.0)*z/self.base_focal
        ty=(cy-self.image_size/2.0)*z/self.base_focal
        return torch.tensor([[tx,ty,z]], dtype=torch.float32, device=self.device)

    def _distance_maps_from_mask(self, mt):
        mn=mt[0,0].detach().cpu().numpy().astype(np.uint8)
        fg=(mn>0).astype(np.uint8)
        do=cv2.distanceTransform(1-fg, cv2.DIST_L2, 5)
        di=cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        do=do/max(float(do.max()), 1e-6); di=di/max(float(di.max()), 1e-6)
        return (
            torch.tensor(do, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0),
            torch.tensor(di, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0),
        )

    def _mask_band_widths(self, mt):
        widths={}
        for name,(y0,y1) in self.BAND_SPECS.items():
            a=max(0, min(self.image_size-1, int(round(y0*self.image_size))))
            b=max(a+1, min(self.image_size, int(round(y1*self.image_size))))
            band=mt[:,:,a:b,:]
            row_mass=band.mean(dim=2).sum(dim=-1)/float(self.image_size)
            widths[name]=row_mass.mean()
        return widths

    def _region_areas(self, mt):
        return {name:(mt*reg).sum()/reg.sum().clamp(min=1.0) for name,reg in self.region_maps.items()}

    def _breast_split_masks(self, mask_tensor, projected_joints, target_mask=None):
        B,C,H,W=mask_tensor.shape
        xs=[]
        for idx in [5,6,11,12]:
            if 0<=idx<projected_joints.shape[1]:
                xs.append(projected_joints[:,idx,0])
        center_x=torch.stack(xs,dim=0).mean(dim=0).view(B,1,1,1) if len(xs)>0 else torch.full((B,1,1,1), float(W)/2.0, device=mask_tensor.device)
        grid_x=torch.arange(W, device=mask_tensor.device, dtype=torch.float32).view(1,1,1,W)
        left_half=(grid_x<center_x).float(); right_half=1.0-left_half
        breast_band=self.region_maps["breast"]
        left_mask=mask_tensor*breast_band*left_half
        right_mask=mask_tensor*breast_band*right_half
        if target_mask is not None:
            left_mask=left_mask*(target_mask>0).float()
            right_mask=right_mask*(target_mask>0).float()
        return left_mask,right_mask,center_x

    def _single_side_metrics(self, side_mask):
        m=side_mask[0,0]
        area=m.mean()
        row_mass=m.sum(dim=0)/float(self.image_size)
        width=row_mass.mean()
        total=m.sum().clamp(min=1e-6)
        ys=torch.arange(self.image_size, device=m.device, dtype=torch.float32).view(self.image_size,1)
        xs=torch.arange(self.image_size, device=m.device, dtype=torch.float32).view(1,self.image_size)
        cx=(m*xs).sum()/total/float(self.image_size)
        cy=(m*ys).sum()/total/float(self.image_size)
        return {"area":area,"width":width,"cx":cx,"cy":cy}

    def _bilateral_target_metrics(self, target_mask, target_joints):
        l,r,_=self._breast_split_masks(target_mask, target_joints, target_mask=target_mask)
        return self._single_side_metrics(l), self._single_side_metrics(r)

    def _bilateral_breast_loss(self, rendered_chest_mask, target_mask, projected_joints, target_joints, metadata):
        cv=float(metadata.get("chest_visible", 0.0))
        if cv<=0.35:
            z=torch.tensor(0.0, device=self.device)
            return z,z,z
        pl,pr,_=self._breast_split_masks(rendered_chest_mask, projected_joints, target_mask=target_mask)
        tl,tr=self._bilateral_target_metrics(target_mask, target_joints)
        plm=self._single_side_metrics(pl); prm=self._single_side_metrics(pr)
        lw=((plm["width"]-tl["width"]).pow(2)+(prm["width"]-tr["width"]).pow(2))*0.5
        la=((plm["area"]-tl["area"]).pow(2)+(prm["area"]-tr["area"]).pow(2))*0.5
        lc=((plm["cx"]-tl["cx"]).pow(2)+(plm["cy"]-tl["cy"]).pow(2)+(prm["cx"]-tr["cx"]).pow(2)+(prm["cy"]-tr["cy"]).pow(2))*0.25
        return lw,la,lc

    def _sternum_cleavage_mask(self, projected_joints):
        B=projected_joints.shape[0]; W=self.image_size; H=self.image_size
        xs=[]
        for idx in [5,6,11,12]:
            if idx<projected_joints.shape[1]:
                xs.append(projected_joints[:,idx,0])
        center_x=torch.stack(xs, dim=0).mean(dim=0).view(B,1,1,1) if len(xs)>0 else torch.full((B,1,1,1), float(W)/2.0, device=self.device)
        grid_x=torch.arange(W, device=self.device, dtype=torch.float32).view(1,1,1,W)
        grid_y=torch.arange(H, device=self.device, dtype=torch.float32).view(1,1,H,1)
        y0=int(0.26*H); y1=int(0.43*H); band_half=0.04*W
        xdist=torch.abs(grid_x-center_x)
        xmask=torch.clamp(1.0-xdist/max(band_half,1.0), min=0.0, max=1.0)
        ymask=((grid_y>=y0)&(grid_y<=y1)).float()
        return xmask*ymask

    def _cleavage_bridge_loss(self, rendered_chest_mask, target_mask, projected_joints, metadata):
        cv=float(metadata.get("chest_visible", 0.0))
        if cv<=0.35:
            return torch.tensor(0.0, device=self.device)
        sm=self._sternum_cleavage_mask(projected_joints)
        pred=(rendered_chest_mask*sm).sum()/sm.sum().clamp(min=1.0)
        tgt=(target_mask*sm).sum()/sm.sum().clamp(min=1.0)
        return F.relu(pred-tgt).pow(2)

    def _sternum_flatten_loss(self, chest_offset_scalars):
        if chest_offset_scalars is None or not self.use_chest_offsets:
            return torch.tensor(0.0, device=self.device)
        scalars=torch.clamp(chest_offset_scalars, -self.chest_offset_limit, self.chest_offset_limit)
        scalars=self._smooth_chest_scalars(scalars)
        verts=self.template_vertices[self.chest_vertex_ids]
        x=verts[:,0]; y=verts[:,1]
        x_mid=x.median(); x_span=(x.max()-x.min()).clamp(min=1e-6)
        y_norm=(y-y.min())/(y.max()-y.min()).clamp(min=1e-6)
        mask=((torch.abs(x-x_mid)<0.10*x_span)&(y_norm>0.20)&(y_norm<0.72))
        if mask.sum()==0:
            return torch.tensor(0.0, device=self.device)
        vals=scalars[:,mask,:]
        return torch.relu(vals).pow(2).mean()


    def _interbreast_gap_loss(self, rendered_body_mask, rendered_chest_mask, target_mask, projected_joints, metadata):
        """
        Stronger version of the cleavage loss.

        The older cleavage loss was too small because it only measured the chest
        render and squared a tiny average. This loss measures both full-body and
        chest overfill in a narrow central band and scales the term so it is
        visible to the optimizer.
        """
        cv=float(metadata.get("chest_visible", 0.0))
        if cv<=0.35:
            return torch.tensor(0.0, device=self.device)

        sm=self._sternum_cleavage_mask(projected_joints)

        # Slightly widen the screen-space band, but keep it restricted to the
        # central sternum / inter-breast zone.
        pooled=F.max_pool2d(sm, kernel_size=9, stride=1, padding=4)
        sm=torch.clamp(0.70*sm+0.30*pooled, 0.0, 1.0)

        pred=torch.clamp(0.65*rendered_body_mask+0.35*rendered_chest_mask, 0.0, 1.0)
        overfill=F.relu(pred-target_mask)*sm

        pred_fill=(pred*sm).sum()/sm.sum().clamp(min=1.0)
        tgt_fill=(target_mask*sm).sum()/sm.sum().clamp(min=1.0)

        # Mean overfill catches local tenting. Fill difference catches the whole
        # central gap being too full. Multipliers intentionally make this term
        # comparable to chest_width/chest_area magnitudes.
        return 25.0*overfill.mean() + 80.0*F.relu(pred_fill-tgt_fill).pow(2)

    def _save_render_debug(
        self,
        debug_dir,
        iteration,
        image_index,
        target_mask,
        rendered_mask,
        chest_mask=None,
        target_joints=None,
        projected_joints=None,
        confidence=None,
        validity=None,
        core_prior=None,
        arm_prior=None,
    ):
        """
        Save overlay and tight-fit diagnostic maps.

        green=target only, red=render only, yellow=overlap,
        blue=detected pose, magenta=projected SMPL-X pose, gray=unknown validity.
        """
        if not self.debug:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        target=target_mask[0,0].detach().float().cpu().numpy()
        render=rendered_mask[0,0].detach().float().cpu().numpy()
        valid_np=np.ones_like(target, dtype=np.float32) if validity is None else validity[0,0].detach().float().cpu().numpy()
        target_bin=(target>0.5).astype(np.uint8)
        render_bin=(render>0.5).astype(np.uint8)
        valid_bin=(valid_np>0.5).astype(np.uint8)
        h,w=target.shape
        img=np.zeros((h,w,3), dtype=np.uint8)
        overlap=(target_bin==1)&(render_bin==1)&(valid_bin==1)
        target_only=(target_bin==1)&(render_bin==0)&(valid_bin==1)
        render_only=(target_bin==0)&(render_bin==1)&(valid_bin==1)
        unknown=(valid_bin==0)
        img[target_only]=[0,255,0]
        img[render_only]=[255,0,0]
        img[overlap]=[255,255,0]
        img[unknown]=[80,80,80]
        if self.debug_keypoints and target_joints is not None and projected_joints is not None:
            tj=target_joints[0].detach().cpu().numpy()
            pj=projected_joints[0].detach().cpu().numpy()
            cf=None if confidence is None else confidence[0].detach().cpu().numpy()
            for a,b in self.COCO_LIMBS:
                if cf is not None and (cf[a] < self.min_keypoint_conf or cf[b] < self.min_keypoint_conf):
                    continue
                pa=tuple(np.round(tj[a]).astype(int)); pb=tuple(np.round(tj[b]).astype(int))
                qa=tuple(np.round(pj[a]).astype(int)); qb=tuple(np.round(pj[b]).astype(int))
                cv2.line(img, pa, pb, (0,128,255), 2)
                cv2.line(img, qa, qb, (255,0,255), 2)
            for j in range(min(17, tj.shape[0])):
                if cf is not None and cf[j] < self.min_keypoint_conf:
                    continue
                tpt=tuple(np.round(tj[j]).astype(int)); ppt=tuple(np.round(pj[j]).astype(int))
                cv2.circle(img, tpt, 4, (0,128,255), -1)
                cv2.circle(img, ppt, 4, (255,0,255), -1)
        prefix=debug_dir / f"iter_{iteration:04d}_img_{image_index:03d}"
        cv2.imwrite(str(prefix)+".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        fp=(render*(1.0-target)*valid_np)
        fn=(target*(1.0-render)*valid_np)
        cv2.imwrite(str(prefix)+"_false_positive_red.png", np.clip(fp*255.0,0,255).astype(np.uint8))
        cv2.imwrite(str(prefix)+"_false_negative_green.png", np.clip(fn*255.0,0,255).astype(np.uint8))
        cv2.imwrite(str(prefix)+"_validity.png", np.clip(valid_np*255.0,0,255).astype(np.uint8))
        if core_prior is not None:
            core_np=core_prior[0,0].detach().float().cpu().numpy()
            core_fp=fp*core_np; core_fn=fn*core_np
            core_target=target*core_np*valid_np
            core_render=render*core_np*valid_np
            cv2.imwrite(str(prefix)+"_core_prior.png", np.clip(core_np*255.0,0,255).astype(np.uint8))
            cv2.imwrite(str(prefix)+"_core_false_positive.png", np.clip(core_fp*255.0,0,255).astype(np.uint8))
            cv2.imwrite(str(prefix)+"_core_false_negative.png", np.clip(core_fn*255.0,0,255).astype(np.uint8))
            cv2.imwrite(str(prefix)+"_pose_core_target.png", np.clip(core_target*255.0,0,255).astype(np.uint8))
            cv2.imwrite(str(prefix)+"_pose_core_rendered.png", np.clip(core_render*255.0,0,255).astype(np.uint8))
        if arm_prior is not None:
            arm_np=arm_prior[0,0].detach().float().cpu().numpy()
            cv2.imwrite(str(prefix)+"_arm_prior.png", np.clip(arm_np*255.0,0,255).astype(np.uint8))

    def _width_loss(self, rm, tw, m):
        rw=self._mask_band_widths(rm)
        loss=torch.tensor(0.0, device=self.device); active=0.0
        cv=float(m.get("chest_visible", 0.0)); hv=float(m.get("hip_visible", 0.0)); ct=m.get("crop_type","")
        if cv>0.40:
            loss+=1.00*(rw["chest"]-tw["chest"]).pow(2)+1.25*(rw["breast"]-tw["breast"]).pow(2)+0.75*(rw["underbust"]-tw["underbust"]).pow(2)
            active+=3.0
        if ct in ["full_body","american"]:
            loss+=0.65*(rw["waist"]-tw["waist"]).pow(2)
            active+=0.65
        if hv>0.40:
            loss+=0.90*(rw["hips"]-tw["hips"]).pow(2)+0.65*(rw["glutes"]-tw["glutes"]).pow(2)
            active+=1.55
        return torch.tensor(0.0, device=self.device) if active<=0 else loss/active

    def _regional_area_loss(self, rm, ta, m):
        ra=self._region_areas(rm)
        loss=torch.tensor(0.0, device=self.device); active=0.0
        cv=float(m.get("chest_visible", 0.0)); hv=float(m.get("hip_visible", 0.0))
        if cv>0.40:
            loss+=self.breast_preserve_weight*(ra["breast"]-ta["breast"]).pow(2)+0.60*(ra["chest"]-ta["chest"]).pow(2)
            active+=self.breast_preserve_weight+0.60
        if hv>0.40:
            loss+=self.glute_preserve_weight*(ra["glutes"]-ta["glutes"]).pow(2)+0.50*(ra["hips"]-ta["hips"]).pow(2)
            active+=self.glute_preserve_weight+0.50
        return torch.tensor(0.0, device=self.device) if active<=0 else loss/active

    def _anti_bloat_loss(self, rm, tm, valid=None):
        if valid is None:
            valid=torch.ones_like(tm)
        denom=valid.mean().clamp(min=1e-6)
        return (rm*(1.0-tm)*self.anti_bloat_map*valid).mean()/denom

    def _projected_chest_loss(self, cm, tm, tw, m):
        cv=float(m.get("chest_visible", 0.0))
        if cv<=0.35:
            z=torch.tensor(0.0, device=self.device)
            return z,z,z
        cp=(0.60*self.region_maps["chest"]+1.00*self.region_maps["breast"]).clamp(0.0,1.0)
        tgt=tm*cp
        fp=cm*(1.0-tm); fn=tgt*(1.0-cm)
        lcs=fp.mean()+0.75*fn.mean()
        cw=self._mask_band_widths(cm)
        lw=(1.50*(cw["breast"]-tw["breast"]).pow(2)+0.90*(cw["chest"]-tw["chest"]).pow(2)+0.60*(cw["underbust"]-tw["underbust"]).pow(2))/3.0
        ca=(cm*self.region_maps["breast"]).sum()/self.region_maps["breast"].sum().clamp(min=1.0)
        ta=(tm*self.region_maps["breast"]).sum()/self.region_maps["breast"].sum().clamp(min=1.0)
        la=(ca-ta).pow(2)
        return lcs,lw,la

    def _abdomen_guard_loss(self, rendered_mask, target_mask, metadata):
        band=0.70*self.region_maps["abdomen"] + 0.30*self.region_maps["waist"]
        band=band.clamp(0.0,1.0)
        pred=(rendered_mask*band).sum()/band.sum().clamp(min=1.0)
        tgt=(target_mask*band).sum()/band.sum().clamp(min=1.0)
        return F.relu(pred-tgt).pow(2)

    def _waist_guard_loss(self, rendered_mask, target_widths, metadata):
        ct=metadata.get("crop_type","")
        if ct not in ["full_body","american"]:
            return torch.tensor(0.0, device=self.device)
        rw=self._mask_band_widths(rendered_mask)
        return (rw["waist"]-target_widths["waist"]).pow(2)

    def _abdomen_area_loss(self, rendered_mask, target_areas, metadata):
        ra=self._region_areas(rendered_mask)
        return (ra["abdomen"]-target_areas["abdomen"]).pow(2)

    def _glute_shape_loss(self, rendered_mask, target_widths, target_areas, metadata):
        hv=float(metadata.get("hip_visible", 0.0))
        if hv<=0.35:
            z=torch.tensor(0.0, device=self.device)
            return z,z,z
        rw=self._mask_band_widths(rendered_mask)
        ra=self._region_areas(rendered_mask)
        width=((rw["hips"]-target_widths["hips"]).pow(2)+(rw["glutes"]-target_widths["glutes"]).pow(2))*0.5
        area=((ra["hips"]-target_areas["hips"]).pow(2)+(ra["glutes"]-target_areas["glutes"]).pow(2))*0.5
        band=(0.55*self.region_maps["hips"]+1.00*self.region_maps["glutes"]).clamp(0.0,1.0)
        pred=(rendered_mask*band).sum()/band.sum().clamp(min=1.0)
        tgt=((target_areas["hips"]+target_areas["glutes"])*0.5).detach()
        bloat=F.relu(pred-tgt).pow(2)
        return width,area,bloat

    def _weighted_keypoint_loss(self, p, t, c):
        c=torch.where(c>self.min_keypoint_conf, c, torch.zeros_like(c))
        w=c*self.joint_weights
        diff=((p-t)/float(self.image_size))**2
        diff=diff.sum(dim=-1)
        denom=w.sum().clamp(min=1.0)
        return (diff*w).sum()/denom

    def _bone_direction_loss(self, p, t, c):
        total=torch.tensor(0.0, device=self.device); count=torch.tensor(0.0, device=self.device)
        for a,b in self.COCO_LIMBS:
            conf=torch.minimum(c[:,a], c[:,b])
            if float(conf.max().detach().cpu())<self.min_keypoint_conf:
                continue
            pv=p[:,b]-p[:,a]; tv=t[:,b]-t[:,a]
            pv=pv/pv.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            tv=tv/tv.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            total+=(((pv-tv)**2).sum(dim=-1)*conf).mean()
            count+=1.0
        return total/count.clamp(min=1.0)

    def _keypoint_center_scale_loss(self, p, t, c):
        c=torch.where(c>self.min_keypoint_conf, c, torch.zeros_like(c))
        w=c*self.joint_weights
        denom=w.sum(dim=1,keepdim=True).clamp(min=1.0)
        pc=(p*w.unsqueeze(-1)).sum(dim=1)/denom
        tc=(t*w.unsqueeze(-1)).sum(dim=1)/denom
        cl=(((pc-tc)/self.image_size)**2).sum(dim=-1).mean()
        ps=torch.sqrt((((p-pc.unsqueeze(1))**2).sum(dim=-1)*w).sum(dim=1)/denom.squeeze(1))
        ts=torch.sqrt((((t-tc.unsqueeze(1))**2).sum(dim=-1)*w).sum(dim=1)/denom.squeeze(1))
        sl=(((ps-ts)/self.image_size)**2).mean()
        return cl+sl

    def _torso_center_scale_loss(self, p, t, c):
        """
        Center/scale loss for torso-only phase.

        Uses shoulders + hips with a local 4-joint weight vector instead of
        the full COCO-17 joint_weights tensor. This fixes shape mismatch in
        the camera_torso phase.
        """
        idx=torch.tensor([5,6,11,12], dtype=torch.long, device=self.device)
        p4=p[:,idx,:]
        t4=t[:,idx,:]
        c4=c[:,idx].clamp(0.0,1.0)
        c4=torch.where(c4>self.min_keypoint_conf, c4, torch.zeros_like(c4))
        local_w=torch.tensor([2.5,2.5,2.5,2.5], dtype=torch.float32, device=self.device).view(1,4)
        w=c4*local_w
        denom=w.sum(dim=1, keepdim=True).clamp(min=1.0)
        pc=(p4*w.unsqueeze(-1)).sum(dim=1)/denom
        tc=(t4*w.unsqueeze(-1)).sum(dim=1)/denom
        cl=(((pc-tc)/self.image_size)**2).sum(dim=-1).mean()
        ps=torch.sqrt((((p4-pc.unsqueeze(1))**2).sum(dim=-1)*w).sum(dim=1)/denom.squeeze(1).clamp(min=1.0))
        ts=torch.sqrt((((t4-tc.unsqueeze(1))**2).sum(dim=-1)*w).sum(dim=1)/denom.squeeze(1).clamp(min=1.0))
        sl=(((ps-ts)/self.image_size)**2).mean()
        return cl+sl


    def _mask_moments(self, mask):
        """
        Differentiable coarse shape statistics from a soft mask.
        """
        m=mask.clamp(0.0,1.0)
        mass=m.sum(dim=(2,3)).clamp(min=1e-6)
        area=mass/float(self.image_size*self.image_size)
        cx=(m*self.grid_x).sum(dim=(2,3))/mass
        cy=(m*self.grid_y).sum(dim=(2,3))/mass
        dx=self.grid_x-cx.view(-1,1,1,1)
        dy=self.grid_y-cy.view(-1,1,1,1)
        vx=(m*dx.pow(2)).sum(dim=(2,3))/mass
        vy=(m*dy.pow(2)).sum(dim=(2,3))/mass
        sx=torch.sqrt(vx.clamp(min=1e-8))
        sy=torch.sqrt(vy.clamp(min=1e-8))
        return {"cx":cx,"cy":cy,"area":area,"sx":sx,"sy":sy}

    def _mask_stats_loss(self, rendered_mask, target_mask, valid=None):
        """
        Coarse alignment loss for center, area and spread.

        If a hair validity mask is available, ignore hair-occluded unknown
        zones for both rendered and target masks so the model is not forced to
        either fill hair or disappear behind hair.
        """
        if valid is not None:
            rendered_mask=rendered_mask*valid
            target_mask=target_mask*valid
        pm=self._mask_moments(rendered_mask)
        tm=self._mask_moments(target_mask)
        center=(pm["cx"]-tm["cx"]).pow(2)+(pm["cy"]-tm["cy"]).pow(2)
        spread=(pm["sx"]-tm["sx"]).pow(2)+(pm["sy"]-tm["sy"]).pow(2)
        area=(pm["area"]-tm["area"]).pow(2)
        return 8.0*center.mean()+4.0*spread.mean()+2.0*area.mean()

    def _torso_keypoint_loss(self, projected, target, confidence):
        idx=torch.tensor([5,6,11,12], dtype=torch.long, device=self.device)
        p=projected[:,idx,:]
        t=target[:,idx,:]
        c=confidence[:,idx].clamp(0.0,1.0)
        c=torch.where(c>self.min_keypoint_conf, c, torch.zeros_like(c))
        diff=((p-t)/float(self.image_size)).pow(2).sum(dim=-1)
        return (diff*c).sum()/c.sum().clamp(min=1.0)

    def _torso_bone_loss(self, projected, target, confidence):
        pairs=[(5,6),(11,12),(5,11),(6,12),(5,12),(6,11)]
        total=torch.tensor(0.0, device=self.device)
        count=torch.tensor(0.0, device=self.device)
        for a,b in pairs:
            conf=torch.minimum(confidence[:,a], confidence[:,b]).clamp(0.0,1.0)
            if float(conf.max().detach().cpu())<self.min_keypoint_conf:
                continue
            pv=projected[:,b]-projected[:,a]
            tv=target[:,b]-target[:,a]
            pv=pv/pv.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            tv=tv/tv.norm(dim=-1,keepdim=True).clamp(min=1e-6)
            total += (((pv-tv).pow(2)).sum(dim=-1)*conf).mean()
            count += 1.0
        return total/count.clamp(min=1.0)

    def _pose_reprojection_pixel_error(self, projected, target, confidence):
        c=confidence.clamp(0.0,1.0)
        c=torch.where(c>self.min_keypoint_conf, c, torch.zeros_like(c))
        e=torch.sqrt(((projected-target).pow(2)).sum(dim=-1).clamp(min=1e-8))
        return (e*c).sum()/c.sum().clamp(min=1.0)


    def _lower_body_keypoint_loss(self, pred, target, conf, image_weight=1.0):
        """
        Stronger lower-body pose lock for hips/knees/ankles.
        Returns:
            loss, lower_reproj_px, pelvis_ctr_loss, thigh_dir_loss
        """
        idx = [11, 12, 13, 14, 15, 16]
        p = pred[:, idx, :]
        t = target[:, idx, :]
        c = conf[:, idx].clamp(0.0, 1.0)
        vis = (c >= self.min_keypoint_conf).float()
        w = c * vis

        diff = torch.norm(p - t, dim=-1)
        denom = w.sum() + 1e-6
        l_kp = (diff * w).sum() / denom

        p_pelvis = 0.5 * (pred[:, 11, :] + pred[:, 12, :])
        t_pelvis = 0.5 * (target[:, 11, :] + target[:, 12, :])
        pelvis_vis = ((conf[:, 11] >= self.min_keypoint_conf) & (conf[:, 12] >= self.min_keypoint_conf)).float()
        pelvis_ctr = (torch.norm(p_pelvis - t_pelvis, dim=-1) * pelvis_vis).sum() / (pelvis_vis.sum() + 1e-6)

        p_center = p.mean(dim=1)
        t_center = t.mean(dim=1)
        lower_ctr = torch.norm(p_center - t_center, dim=-1).mean()

        def _dir(a, b):
            v = b - a
            return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

        p_tl = _dir(pred[:, 11, :], pred[:, 13, :])
        p_tr = _dir(pred[:, 12, :], pred[:, 14, :])
        t_tl = _dir(target[:, 11, :], target[:, 13, :])
        t_tr = _dir(target[:, 12, :], target[:, 14, :])

        vis_tl = ((conf[:, 11] >= self.min_keypoint_conf) & (conf[:, 13] >= self.min_keypoint_conf)).float()
        vis_tr = ((conf[:, 12] >= self.min_keypoint_conf) & (conf[:, 14] >= self.min_keypoint_conf)).float()
        thigh_dir = ((1.0 - (p_tl * t_tl).sum(dim=-1)) * vis_tl + (1.0 - (p_tr * t_tr).sum(dim=-1)) * vis_tr).sum() / (vis_tl.sum() + vis_tr.sum() + 1e-6)

        lower_reproj_px = (diff * vis).sum() / (vis.sum() + 1e-6)

        if self.normalize_lower_body_loss:
            scale=float(max(1, self.image_size))
            l_kp_n=l_kp/scale
            pelvis_ctr_n=pelvis_ctr/scale
            lower_ctr_n=lower_ctr/scale
        else:
            l_kp_n=l_kp
            pelvis_ctr_n=pelvis_ctr
            lower_ctr_n=lower_ctr

        total = image_weight * (
            self.lower_body_kp_weight * l_kp_n
            + self.lower_body_center_weight * 0.5 * (pelvis_ctr_n + lower_ctr_n)
            + self.thigh_direction_weight * thigh_dir
        )
        return total, lower_reproj_px, pelvis_ctr, thigh_dir

    def _draw_capsule_np(self, mask, p0, p1, thickness, value=1.0):
        """Draw a filled capsule into a numpy mask."""
        h,w=mask.shape
        p0=np.asarray(p0, dtype=np.float32)
        p1=np.asarray(p1, dtype=np.float32)
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            return mask
        x0,y0=int(round(p0[0])),int(round(p0[1]))
        x1,y1=int(round(p1[0])),int(round(p1[1]))
        t=max(1,int(round(thickness)))
        cv2.line(mask, (x0,y0), (x1,y1), float(value), thickness=t, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x0,y0), max(1,t//2), float(value), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(mask, (x1,y1), max(1,t//2), float(value), thickness=-1, lineType=cv2.LINE_AA)
        return mask

    def _pose_region_priors(self, target_joints, confidence, target_mask):
        """
        Build pose-aware body-core and arm priors from the target 2D pose.

        Returns:
            core_prior
            arm_prior
            torso_prior
            pelvis_prior
            thigh_prior

        Important:
        - This function returns priors only.
        - The actual optimization target is built later as:
              pose_core_target = pose_core_prior * target_mask * validity

        This avoids the previous issue where the synthetic core prior demanded
        coverage in areas not supported by the real target silhouette.
        """
        H=W=self.image_size
        tj=target_joints[0].detach().cpu().numpy().astype(np.float32)
        cf=confidence[0].detach().cpu().numpy().astype(np.float32)
        tm=target_mask[0,0].detach().cpu().numpy().astype(np.float32)

        torso=np.zeros((H,W), dtype=np.float32)
        pelvis=np.zeros((H,W), dtype=np.float32)
        thigh=np.zeros((H,W), dtype=np.float32)
        arms=np.zeros((H,W), dtype=np.float32)

        def ok(i):
            return 0 <= i < len(cf) and cf[i] >= self.min_keypoint_conf and np.all(np.isfinite(tj[i]))

        # Robust local body scale from torso and upper-leg geometry.
        lengths=[]
        for a,b in [(5,6),(11,12),(5,11),(6,12),(11,13),(12,14)]:
            if ok(a) and ok(b):
                lengths.append(float(np.linalg.norm(tj[a]-tj[b])))

        scale=max(18.0, float(np.median(lengths)) if lengths else float(H)*0.18)

        torso_pad=max(7, int(0.085*scale))

        # Weaken upper-thigh capsules:
        # previous: thickness ~= 0.22*scale, value 0.85
        # now:      thickness ~= 0.14*scale, value 0.45
        # This keeps thighs represented but stops them dominating core coverage.
        thigh_thick=max(7, int(0.14*scale))
        arm_thick=max(8, int(0.15*scale))

        if ok(5) and ok(6) and ok(11) and ok(12):
            ls,rs,lh,rh=tj[5],tj[6],tj[11],tj[12]
            shoulder_vec=rs-ls
            hip_vec=rh-lh

            if np.linalg.norm(shoulder_vec)>1e-6:
                sv=shoulder_vec/np.linalg.norm(shoulder_vec)
                ls2=ls-sv*torso_pad
                rs2=rs+sv*torso_pad
            else:
                ls2,rs2=ls,rs

            if np.linalg.norm(hip_vec)>1e-6:
                hv=hip_vec/np.linalg.norm(hip_vec)
                lh2=lh-hv*torso_pad
                rh2=rh+hv*torso_pad
            else:
                lh2,rh2=lh,rh

            torso_poly=np.array([ls2,rs2,rh2,lh2], dtype=np.int32).reshape(-1,1,2)
            cv2.fillPoly(torso, [torso_poly], 1.00, lineType=cv2.LINE_AA)

            pelvis_center=(lh+rh)*0.5
            chest_center=(ls+rs)*0.5
            abdomen_center=(0.58*pelvis_center+0.42*chest_center)
            hip_width=max(12.0, float(np.linalg.norm(hip_vec)))
            torso_height=max(12.0, float(np.linalg.norm(pelvis_center-chest_center)))
            axes=(max(10,int(hip_width*0.50)), max(12,int(torso_height*0.22)))
            cv2.ellipse(
                pelvis,
                tuple(np.round(abdomen_center).astype(int)),
                axes,
                0,
                0,
                360,
                1.0,
                -1,
                lineType=cv2.LINE_AA,
            )

        # Weak upper-thigh priors. Do not include lower legs in body-core fitting.
        if ok(11) and ok(13):
            self._draw_capsule_np(thigh, tj[11], tj[13], thigh_thick, 0.45)
        if ok(12) and ok(14):
            self._draw_capsule_np(thigh, tj[12], tj[14], thigh_thick, 0.45)

        # Arm prior reduces silhouette pressure around raised arms.
        for a,b in [(5,7),(7,9),(6,8),(8,10)]:
            if ok(a) and ok(b):
                self._draw_capsule_np(arms, tj[a], tj[b], arm_thick, 1.0)

        # Restrict priors to a dilated target-neighborhood. The final core target
        # is additionally multiplied by target_mask and validity in optimize().
        target_bin=(tm>0.5).astype(np.uint8)
        k=max(3, int(self.pose_core_dilate_px) | 1)
        target_dil=cv2.dilate(
            target_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),
            iterations=2,
        ).astype(np.float32)

        torso*=target_dil
        pelvis*=target_dil
        thigh*=target_dil

        arm_dil=cv2.dilate(
            target_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),
            iterations=1,
        ).astype(np.float32)
        arms*=arm_dil

        sigma=max(1.0, k*0.35)
        torso=cv2.GaussianBlur(torso, (0,0), sigmaX=sigma)
        pelvis=cv2.GaussianBlur(pelvis, (0,0), sigmaX=sigma)
        thigh=cv2.GaussianBlur(thigh, (0,0), sigmaX=sigma)
        arms=cv2.GaussianBlur(arms, (0,0), sigmaX=sigma)

        torso=np.clip(torso,0.0,1.0).astype(np.float32)
        pelvis=np.clip(pelvis,0.0,1.0).astype(np.float32)
        thigh=np.clip(thigh,0.0,1.0).astype(np.float32)
        arms=np.clip(arms,0.0,1.0).astype(np.float32)

        # Weaken thighs in the combined core prior.
        core=np.clip(1.00*torso + 0.90*pelvis + 0.35*thigh, 0.0, 1.0).astype(np.float32)

        core_t=torch.tensor(core, dtype=torch.float32, device=self.device).view(1,1,H,W)
        arm_t=torch.tensor(arms, dtype=torch.float32, device=self.device).view(1,1,H,W)
        torso_t=torch.tensor(torso, dtype=torch.float32, device=self.device).view(1,1,H,W)
        pelvis_t=torch.tensor(pelvis, dtype=torch.float32, device=self.device).view(1,1,H,W)
        thigh_t=torch.tensor(thigh, dtype=torch.float32, device=self.device).view(1,1,H,W)

        return core_t, arm_t, torso_t, pelvis_t, thigh_t

    def _tight_fit_losses(self, rendered_mask, target_mask, valid=None):
        """
        Adaptive full-silhouette FP/FN loss.

        This is the global complement to the pose-aware core loss. It reduces
        red overfill without blindly shrinking the model when green missing
        target area is larger than false-positive overfill.

        Returns:
            tight:    balanced adaptive FP/FN loss
            fp:       rendered outside target
            fn:       target missed by render
            outside:  outside-target loss weighted by anti_bloat_map
        """
        if valid is None:
            valid=torch.ones_like(target_mask)

        v=valid.clamp(0.0,1.0)
        denom=v.sum().clamp(min=1.0)

        fp_map=rendered_mask*(1.0-target_mask)*v
        fn_map=target_mask*(1.0-rendered_mask)*v

        fp=fp_map.sum()/denom
        fn=fn_map.sum()/denom

        # Adaptive balancing: when FN is larger than FP, increase coverage
        # pressure. When FP is larger than FN, increase outside pressure.
        # Detach the ratios so the optimizer follows the losses rather than
        # gaming the weighting function itself.
        fn_w=torch.clamp((fn.detach()+1e-6)/(fp.detach()+1e-6), 0.80, 2.75)
        fp_w=torch.clamp((fp.detach()+1e-6)/(fn.detach()+1e-6), 0.80, 2.25)

        tight=fp_w*fp + fn_w*fn

        # Anti-bloat-weighted outside target pressure. This stays moderate in
        # the final schedules because core coverage can otherwise collapse.
        outside=(fp_map*self.anti_bloat_map).sum()/denom

        return tight, fp, fn, outside

    def _core_tight_fit_losses(self, rendered_mask, core_target, core_weight=None):
        """
        Pose-aware body-core FP/FN losses.

        Args:
            rendered_mask:
                rendered full-body silhouette.
            core_target:
                already gated target:
                    pose_core_prior * target_mask * validity
            core_weight:
                soft spatial weight for the region, usually:
                    pose_core_prior * validity

        This implements the intended behavior explicitly instead of relying on
        internal multiplication of target/validity/prior.
        """
        if core_weight is None:
            core_weight=(core_target>0).float()

        w=core_weight.clamp(0.0,1.0)
        tgt=core_target.clamp(0.0,1.0)
        denom=w.sum().clamp(min=1.0)

        fp=(rendered_mask*(1.0-tgt)*w).sum()/denom
        fn=(tgt*(1.0-rendered_mask)).sum()/denom

        cov_w=torch.clamp(
            (fn.detach()+1e-6)/(fp.detach()+1e-6),
            1.0,
            self.core_coverage_adaptive_max,
        )
        coverage=cov_w*fn

        return fp, fn, coverage

    def _distance_silhouette_loss(self, rm, tm, do, di, valid=None):
        if valid is None:
            valid=torch.ones_like(tm)
        denom=valid.mean().clamp(min=1e-6)
        fp=rm*(1.0-tm)*do*valid
        fn=tm*(1.0-rm)*di*valid
        return (fp.mean()+0.30*fn.mean())/denom

    def _iou_loss(self, rm, tm, valid=None, eps=1e-6):
        if valid is None:
            valid=torch.ones_like(tm)
        inter=(rm*tm*valid).sum(dim=(1,2,3))
        union=((rm+tm-rm*tm)*valid).sum(dim=(1,2,3))
        return 1.0-((inter+eps)/(union+eps)).mean()

    def _edge_loss(self, rm, tm, valid=None):
        if valid is None:
            valid=torch.ones_like(tm)

        # Suppress edge loss in unknown hair-occluded zones.
        rm=rm*valid
        tm=tm*valid

        kx=torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]],dtype=torch.float32,device=self.device).unsqueeze(0)
        ky=torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]],dtype=torch.float32,device=self.device).unsqueeze(0)
        px,py=F.conv2d(rm,kx,padding=1),F.conv2d(rm,ky,padding=1)
        tx,ty=F.conv2d(tm,kx,padding=1),F.conv2d(tm,ky,padding=1)
        pe=torch.sqrt(px**2+py**2+1e-6); te=torch.sqrt(tx**2+ty**2+1e-6)
        denom=valid.mean().clamp(min=1e-6)
        return torch.abs(pe-te).mean()/denom


    def _smoothstep(self, x):
        x=float(np.clip(x, 0.0, 1.0))
        return x*x*(3.0-2.0*x)

    def _anchor_losses(
        self,
        pj,
        rm,
        betas,
        bp_i,
        go_i,
        tr_i,
        lfs,
        anchor,
        image_index,
        sil_loss,
        outside_loss,
        iou_loss_value,
        valid,
    ):
        """
        Constrain late-stage shape/chest fitting so it cannot destroy the
        already-good pose/camera/silhouette alignment found in earlier phases.

        This is not best-state tracking. It is a differentiable trust-region.
        """
        if anchor is None:
            z=torch.tensor(0.0, device=self.device)
            return z,z,z,z,z

        aj=anchor["projected_joints"][image_index]
        am=anchor["rendered_masks"][image_index]
        abp=anchor["body_pose"][image_index]
        ago=anchor["global_orient"][image_index]
        atr=anchor["translation"][image_index]

        # Projected-joint anchor prevents reprojection drift after the pose is good.
        joint_anchor=(((pj-aj)/float(self.image_size))**2).mean()

        pose_anchor=((bp_i-abp)**2).mean() + 0.25*((go_i-ago)**2).mean()

        camera_anchor=((tr_i-atr)**2).mean()
        if lfs is not None and anchor.get("focal_log_scale", None) is not None:
            camera_anchor=camera_anchor + 0.25*((lfs-anchor["focal_log_scale"])**2).mean()

        beta_anchor=((betas-anchor["betas"])**2).mean()

        # Monotonic guard: do not let late phases worsen key silhouette scalars.
        anchor_sil=anchor["sil"][image_index]
        anchor_out=anchor["outside"][image_index]
        anchor_iou=anchor["iou"][image_index]

        sil_guard=F.relu(sil_loss - anchor_sil - self.anchor_sil_tolerance).pow(2)
        out_guard=F.relu(outside_loss - anchor_out - self.anchor_outside_tolerance).pow(2)
        iou_guard=F.relu(iou_loss_value - anchor_iou - self.anchor_iou_tolerance).pow(2)

        # Also block new rendered overfill relative to the anchor silhouette.
        # This catches broad torso/chest expansion even if target scalar losses are noisy.
        anchor_over=(rm*(1.0-am)*valid).mean()

        silhouette_guard=sil_guard + 2.0*out_guard + iou_guard + 0.5*anchor_over

        return joint_anchor, pose_anchor, camera_anchor, beta_anchor, silhouette_guard

    def _chest_residual_loss(self, chest_mask, target_mask, anchor_mask, valid):
        """
        Late chest offsets should explain only local missing breast/chest
        residuals from the anchored body fit, while avoiding new outside pixels.
        """
        breast_region=(0.65*self.region_maps["breast"]+0.35*self.region_maps["chest"]).clamp(0.0,1.0)
        pos_residual=F.relu(target_mask-anchor_mask)*breast_region*valid
        coverage=(pos_residual*(1.0-chest_mask)).mean()
        outside=(chest_mask*(1.0-target_mask)*breast_region*valid).mean()
        return coverage + self.chest_outside_guard_weight*outside


    def _evaluate_anchor_candidate(self, betas, cos, bp, go, tr, lfs, cc, masks, valids, num_images):
        """
        Evaluate the current state before locking constrained refinement.
        Returns metrics and the fully built anchor payload.
        """
        with torch.no_grad():
            cof_anchor=self._chest_offsets_full(cos)
            anchor={
                "betas":betas.detach().clone(),
                "body_pose":[p.detach().clone() for p in bp],
                "global_orient":[p.detach().clone() for p in go],
                "translation":[p.detach().clone() for p in tr],
                "focal_log_scale":None if lfs is None else lfs.detach().clone(),
                "projected_joints":[],
                "rendered_masks":[],
                "sil":[],
                "outside":[],
                "iou":[],
            }
            sil_vals=[]; outside_vals=[]; iou_vals=[]; reproj_vals=[]
            rw0=BodyRegionWeights.create_weight_map(self.image_size,self.image_size,self.device)
            if rw0.dim()==3:
                rw0=rw0.unsqueeze(0)

            # If target joints are not in scope, this function receives them via
            # attributes set by optimize() just before calling.
            gtk=getattr(self, "_anchor_eval_gtk", None)
            conf=getattr(self, "_anchor_eval_conf", None)

            for ai in range(num_images):
                fl_anchor=self._current_focal(lfs, ai)
                aout=self.model(betas=betas, body_pose=bp[ai], global_orient=go[ai], transl=None, return_verts=True)
                averts=aout.vertices + cof_anchor
                ajoints=SMPLXJointMapper.smplx_to_coco17(aout.joints)
                apj=self._project_points_screen(ajoints, f=fl_anchor, cc=cc, t=tr[ai])
                arend=self.renderer.render(vertices=averts, faces=self.model.faces_tensor.unsqueeze(0), focal_length=fl_anchor, principal_point=cc, translation=tr[ai])
                arm=arend[...,3].unsqueeze(1).detach()
                valid_anchor=valids[ai]
                asil=silhouette_loss(arm, masks[ai], valid_anchor, rw0)
                _,_,_,aoutside=self._tight_fit_losses(arm, masks[ai], valid_anchor)
                aiou=self._iou_loss(arm, masks[ai], valid_anchor)

                if gtk is not None and conf is not None:
                    arpx=self._pose_reprojection_pixel_error(apj, gtk[ai], conf[ai])
                else:
                    arpx=torch.tensor(999.0, device=self.device)

                anchor["projected_joints"].append(apj.detach().clone())
                anchor["rendered_masks"].append(arm.detach().clone())
                anchor["sil"].append(asil.detach())
                anchor["outside"].append(aoutside.detach())
                anchor["iou"].append(aiou.detach())
                sil_vals.append(float(asil.detach().item()))
                outside_vals.append(float(aoutside.detach().item()))
                iou_vals.append(float(aiou.detach().item()))
                reproj_vals.append(float(arpx.detach().item()))

            metrics={
                "mean_reproj":float(np.mean(reproj_vals)),
                "mean_sil":float(np.mean(sil_vals)),
                "mean_outside":float(np.mean(outside_vals)),
                "mean_iou":float(np.mean(iou_vals)),
            }
            return metrics, anchor

    def _anchor_is_good_enough(self, metrics):
        return (
            metrics["mean_reproj"] <= self.anchor_reproj_threshold and
            metrics["mean_sil"] <= self.anchor_sil_threshold and
            metrics["mean_outside"] <= self.anchor_outside_threshold
        )

    def _delay_refinement_cfg(self, cfg):
        """
        If no good anchor exists yet, do not enter frozen region/chest refinement.
        Continue a pose+body-size phase with pose/camera trainable.
        """
        if cfg["name"] not in ["region_shape_no_chest_offsets", "final_constrained_chest", "final_adaptive_tight_chest"]:
            return cfg
        delayed=dict(cfg)
        delayed["name"]="body_size_shape_wait_anchor"
        delayed["train_betas"]=True
        delayed["train_chest"]=False
        delayed["train_pose"]=True
        delayed["train_orient"]=True
        delayed["train_trans"]=True
        delayed["train_focal"]=True
        delayed["lr"]=min(float(cfg.get("lr",0.006)), 0.006)
        # Turn off chest-local losses while waiting for a strong anchor.
        for k in ["chest_proj","chest_width","chest_area","bilat_w","bilat_a","bilat_c","cleavage","sternum","gap","chest_reg","chest_residual"]:
            delayed[k]=0.0
        # Keep shape/silhouette active but moderate.
        delayed["anchor_joint"]=0.0
        delayed["anchor_pose"]=0.0
        delayed["anchor_camera"]=0.0
        delayed["anchor_beta"]=0.0
        delayed["sil_guard"]=0.0
        delayed["pose"]=0.014
        delayed["trans"]=0.045
        delayed["focal_reg"]=0.035
        return delayed

    def _phase_config(self, it, its):
        """
        Constrained hierarchical schedule.

        Key behavior:
        - pose/camera fit first
        - body-size shape fits with anchors
        - region phase is pose/camera frozen
        - final chest phase is local: pose/camera/betas frozen, chest offsets only
        """
        f=it/max(1, its-1)

        def ramp(a,b):
            return self._smoothstep((f-a)/max(1e-6,b-a))

        base_zero_regions={
            "width":0.00,"area":0.00,"bloat":0.00,
            "tight":0.00,"outside":0.00,"core_fp":0.00,"core_fn":0.00,"core_cov":0.00,"arm_reduce":0.00,
            "chest_proj":0.00,"chest_width":0.00,"chest_area":0.00,
            "bilat_w":0.00,"bilat_a":0.00,"bilat_c":0.00,
            "cleavage":0.00,"sternum":0.00,"gap":0.00,
            "abdomen":0.00,"abdomen_area":0.00,"waist":0.00,
            "glute_w":0.00,"glute_a":0.00,"glute_bloat":0.00,
            "chest_reg":0.00,
            "anchor_joint":0.00,"anchor_pose":0.00,"anchor_camera":0.00,"anchor_beta":0.00,"sil_guard":0.00,
            "chest_residual":0.00,
            "lb_kp":0.00,"lb_ctr":0.00,"lb_dir":0.00,"lb_reproj":0.00,
        }

        if f<0.25:
            cfg={"name":"camera_torso","kp":120.0,"bone":50.0,"center":80.0,
                 "mask_stats":18.0*self.mask_stats_weight,"sil":0.03,"dist":0.40,"iou":0.15,"edge":0.00,
                 "shape":1.00,"beta":0.75,"pose":0.010,"trans":0.08,"focal_reg":0.12,
                 "train_betas":False,"train_chest":False,"train_pose":True,"train_orient":True,"train_trans":True,"train_focal":True,
                 "torso_only":True,"lr":0.025}
            cfg.update(base_zero_regions); return cfg

        if f<0.65:
            cfg={"name":"pose_lock","kp":105.0*self.pose_lock_weight,"bone":50.0*self.pose_lock_weight,"center":50.0,
                 "mask_stats":12.0*self.mask_stats_weight,"sil":0.08,"dist":1.40,"iou":0.40,"edge":0.02,
                 "tight":0.20*self.tight_fit_weight,"outside":0.05*self.outside_target_weight,
                 "core_fp":0.05,"core_fn":0.20*self.core_coverage_weight,"core_cov":0.20*self.core_coverage_weight,
                 "arm_reduce":0.20*self.arm_silhouette_reduction,
                 "lb_kp":2.30,"lb_ctr":1.40,"lb_dir":1.10,"lb_reproj":0.12,
                 "shape":0.90,"beta":0.65,"pose":0.010,"trans":0.06,"focal_reg":0.06,
                 "train_betas":False,"train_chest":False,"train_pose":True,"train_orient":True,"train_trans":True,"train_focal":True,
                 "torso_only":False,"lr":0.018}
            cfg.update({k:v for k,v in base_zero_regions.items() if k not in cfg}); return cfg

        if f<0.86:
            r=ramp(0.65,0.86)
            return {"name":"body_size_shape","kp":70.0,"bone":34.0,"center":30.0,
                    "mask_stats":8.0*self.mask_stats_weight,"sil":0.16,"dist":3.50,"iou":0.75,"edge":0.05,
                    "tight":(0.80+0.40*r)*self.tight_fit_weight,
                    "outside":(0.45+0.20*r)*self.outside_target_weight,
                    "core_fp":0.50+0.20*r,
                    "core_fn":(1.00+0.30*r)*self.core_coverage_weight,
                    "core_cov":(1.50+0.60*r)*self.core_coverage_weight,
                    "arm_reduce":0.50*self.arm_silhouette_reduction,
                    "width":self.width_weight*(0.25+0.25*r),
                    "area":self.area_weight*(0.20+0.20*r),
                    "bloat":self.anti_bloat_weight*(0.35+0.25*r),
                    "chest_proj":0.00,"chest_width":0.00,"chest_area":0.00,
                    "bilat_w":0.00,"bilat_a":0.00,"bilat_c":0.00,"cleavage":0.00,"sternum":0.00,"gap":0.00,
                    "abdomen":self.abdomen_guard_weight*0.35*r,
                    "abdomen_area":self.abdomen_area_weight*0.35*r,
                    "waist":self.waist_guard_weight*0.35*r,
                    "glute_w":self.glute_width_weight*0.35*r,
                    "glute_a":self.glute_area_weight*0.35*r,
                    "glute_bloat":self.glute_bloat_weight*0.35*r,
                    "anchor_joint":self.joint_anchor_weight*0.35,
                    "anchor_pose":self.pose_anchor_weight*0.35,
                    "anchor_camera":self.camera_anchor_weight*0.35,
                    "anchor_beta":self.beta_anchor_weight*0.50,
                    "sil_guard":self.silhouette_guard_weight*0.35,
                    "chest_residual":0.00,
                    "lb_kp":1.50,"lb_ctr":1.00,"lb_dir":0.80,"lb_reproj":0.08,
                    "chest_reg":0.00,"shape":0.30,"beta":0.35,"pose":0.015,"trans":0.05,"focal_reg":0.035,
                    "train_betas":True,"train_chest":False,"train_pose":True,"train_orient":True,"train_trans":True,"train_focal":True,
                    "torso_only":False,"lr":0.010}

        if f<0.95:
            r=ramp(0.86,0.95)
            return {"name":"region_shape_no_chest_offsets","kp":70.0,"bone":34.0,"center":30.0,
                    "mask_stats":6.0*self.mask_stats_weight,"sil":0.20,"dist":4.80,"iou":0.95,"edge":0.06,
                    "tight":(0.95+0.35*r)*self.tight_fit_weight,
                    "outside":(0.55+0.20*r)*self.outside_target_weight,
                    "core_fp":0.55,
                    "core_fn":1.20*self.core_coverage_weight,
                    "core_cov":1.80*self.core_coverage_weight,
                    "arm_reduce":0.75*self.arm_silhouette_reduction,
                    "width":self.width_weight*(0.40+0.25*r),
                    "area":self.area_weight*(0.30+0.25*r),
                    "bloat":self.anti_bloat_weight*0.75,
                    "chest_proj":self.chest_project_weight*0.20*r,
                    "chest_width":self.chest_width_weight*0.20*r,
                    "chest_area":self.chest_area_weight*0.15*r,
                    "bilat_w":0.00,"bilat_a":0.00,"bilat_c":0.00,
                    "cleavage":0.50*r,"sternum":0.50*r,"gap":1.00*r,
                    "abdomen":self.abdomen_guard_weight*0.75,
                    "abdomen_area":self.abdomen_area_weight*0.75,
                    "waist":self.waist_guard_weight*0.75,
                    "glute_w":self.glute_width_weight*0.75,
                    "glute_a":self.glute_area_weight*0.75,
                    "glute_bloat":self.glute_bloat_weight*0.75,
                    "anchor_joint":self.joint_anchor_weight,
                    "anchor_pose":self.pose_anchor_weight,
                    "anchor_camera":self.camera_anchor_weight,
                    "anchor_beta":self.beta_anchor_weight*1.50,
                    "sil_guard":self.silhouette_guard_weight,
                    "chest_residual":0.00,
                    "lb_kp":1.00,"lb_ctr":0.60,"lb_dir":0.50,"lb_reproj":0.05,
                    "chest_reg":0.00,"shape":0.18,"beta":0.40,"pose":0.020,"trans":0.06,"focal_reg":0.04,
                    "train_betas":True,"train_chest":False,
                    # Critical: freeze pose/camera in region phase so shape cannot
                    # damage a good reprojection solution.
                    "train_pose":False,"train_orient":False,"train_trans":False,"train_focal":False,
                    "torso_only":False,"lr":0.006}

        r=ramp(0.95,1.00)
        return {"name":"final_constrained_chest","kp":70.0,"bone":34.0,"center":30.0,
                "mask_stats":4.0*self.mask_stats_weight,"sil":0.18,"dist":4.80,"iou":0.90,"edge":0.06,
                "tight":1.10*self.tight_fit_weight,
                "outside":0.70*self.outside_target_weight,
                "core_fp":0.35,
                "core_fn":0.75*self.core_coverage_weight,
                "core_cov":1.20*self.core_coverage_weight,
                "arm_reduce":0.85*self.arm_silhouette_reduction,
                "width":self.width_weight*0.35,
                "area":self.area_weight*0.35,
                "bloat":self.anti_bloat_weight*0.80,
                # Chest refinement only, ramped in smoothly and locally.
                "chest_proj":self.chest_project_weight*0.45*r,
                "chest_width":self.chest_width_weight*0.45*r,
                "chest_area":self.chest_area_weight*0.35*r,
                "bilat_w":self.bilateral_breast_weight*0.60*r,
                "bilat_a":self.bilateral_breast_weight*0.50*r,
                "bilat_c":self.bilateral_centroid_weight*0.50*r,
                "cleavage":2.50*r,"sternum":1.50*r,"gap":4.00*r,
                "abdomen":self.abdomen_guard_weight*0.55,
                "abdomen_area":self.abdomen_area_weight*0.55,
                "waist":self.waist_guard_weight*0.55,
                "glute_w":self.glute_width_weight*0.55,
                "glute_a":self.glute_area_weight*0.55,
                "glute_bloat":self.glute_bloat_weight*0.55,
                "anchor_joint":self.joint_anchor_weight*1.20,
                "anchor_pose":self.pose_anchor_weight*1.20,
                "anchor_camera":self.camera_anchor_weight*1.20,
                "anchor_beta":self.beta_anchor_weight*3.00,
                "sil_guard":self.silhouette_guard_weight*1.30,
                "chest_residual":self.chest_residual_weight*r,
                "lb_kp":0.80,"lb_ctr":0.50,"lb_dir":0.40,"lb_reproj":0.04,
                "chest_reg":self.chest_offset_weight*1.20,
                "shape":0.10,"beta":0.50,"pose":0.030,"trans":0.08,"focal_reg":0.05,
                # Critical final phase: local chest offsets only.
                "train_betas":False,"train_chest":True,
                "train_pose":False,"train_orient":False,"train_trans":False,"train_focal":False,
                "torso_only":False,"lr":0.004}

    def _set_trainable(self, params, flag):
        for p in params: p.requires_grad_(flag)

    def _make_optimizer(self, betas, cos, bp, go, tr, lfs, cfg):
        betas.requires_grad_(cfg["train_betas"])
        if cos is not None: cos.requires_grad_(cfg["train_chest"])
        self._set_trainable(bp.parameters(), cfg["train_pose"])
        self._set_trainable(go.parameters(), cfg["train_orient"])
        self._set_trainable(tr.parameters(), cfg["train_trans"])
        if lfs is not None: lfs.requires_grad_(cfg["train_focal"])
        params=[]
        if betas.requires_grad: params.append({"params":[betas],"lr":cfg["lr"]*0.20})
        if cos is not None and cos.requires_grad: params.append({"params":[cos],"lr":cfg["lr"]*0.35})
        pp=[p for p in bp.parameters() if p.requires_grad]
        op=[p for p in go.parameters() if p.requires_grad]
        tp=[p for p in tr.parameters() if p.requires_grad]
        if pp: params.append({"params":pp,"lr":cfg["lr"]})
        if op: params.append({"params":op,"lr":cfg["lr"]*0.5})
        if tp: params.append({"params":tp,"lr":cfg["lr"]*0.5})
        if lfs is not None and lfs.requires_grad: params.append({"params":[lfs],"lr":cfg["lr"]*0.1})
        if not params: params=[{"params":[tr[0]],"lr":cfg["lr"]}]
        return torch.optim.Adam(params)

    def optimize(self, image_paths, pose_json_paths, visibility_json_paths, output_path, iterations=1000):
        print("\n🚀 Starting chest/abdomen/glute-aware multi-image optimization")
        output_path=Path(output_path)
        debug_dir=output_path.parent / "_render_debug"
        num_images=len(image_paths)
        print(f"✓ Images: {num_images}")

        betas=nn.Parameter(torch.zeros(1,10,device=self.device))
        initial_betas=torch.zeros_like(betas).detach()
        cos=nn.Parameter(torch.zeros(1,int(self.chest_vertex_ids.numel()),1,dtype=torch.float32,device=self.device)) if self.use_chest_offsets else None
        lfs=nn.Parameter(torch.zeros(num_images,device=self.device)) if self.optimize_focal else None

        bp,go,tr=nn.ParameterList(),nn.ParameterList(),nn.ParameterList()
        gtk,conf,masks,valids=[],[],[],[]
        do_all,di_all,meta,tw_all,ta_all,iw=[],[],[],[],[],[]
        validity_mean_all=[]
        pose_quality_all=[]
        pose_image_weight_all=[]
        pose_keypoint_weight_all=[]
        pose_backend_all=[]
        pose_model_all=[]

        for i in range(num_images):
            pose_data_full=self._load_pose_json_full(pose_json_paths[i])
            pose_info=self._pose_quality_info(pose_data_full)
            kn=self._load_scaled_pose(pose_json_paths[i], image_paths[i])

            md=load_visibility_json(visibility_json_paths[i])
            md["pose_quality_score"]=pose_info["quality"]
            md["pose_image_weight"]=pose_info["image_weight"]
            md["pose_keypoint_weight"]=pose_info["keypoint_weight"]
            md["pose_low_quality"]=pose_info["is_low"]
            md["pose_very_low_quality"]=pose_info["is_very_low"]
            md["pose_alpha_support"]=pose_info["support"]
            md["pose_sanity_score"]=pose_info["sanity"]
            md["pose_backend"]=pose_info["backend"]
            md["pose_model"]=pose_info["model"]
            meta.append(md)

            pose_quality_all.append(pose_info["quality"])
            pose_image_weight_all.append(pose_info["image_weight"])
            pose_keypoint_weight_all.append(pose_info["keypoint_weight"])
            pose_backend_all.append(pose_info["backend"])
            pose_model_all.append(pose_info["model"])

            gtk.append(torch.tensor(kn[:,:2], dtype=torch.float32, device=self.device).unsqueeze(0))

            # Keep raw joint confidence values, but suppress all keypoint-based
            # losses for bad pose estimates through pose_keypoint_weight_all.
            conf.append(torch.tensor(kn[:,2], dtype=torch.float32, device=self.device).unsqueeze(0))

            mt=self._load_target_mask(image_paths[i]); masks.append(mt)
            vt=self._load_silhouette_validity(image_paths[i]); valids.append(vt)
            validity_mean_all.append(self._validity_stats(vt))
            do,di=self._distance_maps_from_mask(mt); do_all.append(do); di_all.append(di)
            tw_all.append(self._mask_band_widths(mt)); ta_all.append(self._region_areas(mt))

            # Mild full-image downweight from pose metadata. The silhouette and
            # alpha mask still matter, so this is intentionally not as strong as
            # the keypoint-specific downweight.
            iw.append(float(md.get("image_weight", 1.0))*float(pose_info["image_weight"]))

            mn=mt[0,0].detach().cpu().numpy().astype(np.float32)
            bp.append(nn.Parameter(torch.zeros(1,63,device=self.device)))
            go.append(nn.Parameter(torch.zeros(1,3,device=self.device)))
            tr.append(nn.Parameter(self._initial_translation_from_mask_and_keypoints(mn, kn)))

        mean_w=max(float(np.mean(iw)), 1e-6)
        iw=[float(np.clip(w/mean_w, 0.25, 2.0)) for w in iw]

        if len(pose_quality_all)>0:
            print(
                f"✓ Pose quality: mean={np.mean(pose_quality_all):.3f}, "
                f"min={np.min(pose_quality_all):.3f}, "
                f"kp_weight_mean={np.mean(pose_keypoint_weight_all):.3f}"
            )
            low_count=sum(1 for q in pose_quality_all if q < self.pose_quality_low_threshold)
            very_low_count=sum(1 for q in pose_quality_all if q < self.pose_quality_very_low_threshold)
            if low_count>0 or very_low_count>0:
                print(f"⚠ Pose downweighting: low={low_count}, very_low={very_low_count}")

        if len(validity_mean_all)>0 and self.use_hair_aware_masks:
            print(
                f"✓ Hair-aware silhouette validity: mean={np.mean(validity_mean_all):.3f}, "
                f"min={np.min(validity_mean_all):.3f}"
            )

        cc=torch.tensor([[self.image_size/2.0, self.image_size/2.0]], device=self.device)

        opt=None; phase=None
        anchor=None
        prog=tqdm(range(iterations), desc="Optimizing", dynamic_ncols=True, leave=True)

        for it in prog:
            cfg=self._phase_config(it, iterations)
            if phase != cfg["name"]:
                phase=cfg["name"]

                # Capture only a sufficiently good pose/camera/body-scale anchor.
                # If it is not good yet, keep pose/camera trainable and delay
                # region/chest refinement instead of locking a mediocre state.
                if anchor is None and cfg["name"] in ["body_size_shape","region_shape_no_chest_offsets","final_constrained_chest","final_adaptive_tight_chest"]:
                    self._anchor_eval_gtk=gtk
                    self._anchor_eval_conf=conf
                    anchor_metrics, candidate_anchor = self._evaluate_anchor_candidate(
                        betas=betas, cos=cos, bp=bp, go=go, tr=tr, lfs=lfs, cc=cc,
                        masks=masks, valids=valids, num_images=num_images
                    )
                    if self._anchor_is_good_enough(anchor_metrics):
                        anchor=candidate_anchor
                        tqdm.write(
                            f"🔒 Captured good constrained-refinement anchor at iter {it:04d} / phase={cfg['name']} "
                            f"(mean_reproj={anchor_metrics['mean_reproj']:.2f}, "
                            f"mean_sil={anchor_metrics['mean_sil']:.4f}, "
                            f"mean_out={anchor_metrics['mean_outside']:.4f})"
                        )
                    else:
                        tqdm.write(
                            f"⏳ Delaying region/chest refinement at iter {it:04d}: "
                            f"mean_reproj={anchor_metrics['mean_reproj']:.2f} "
                            f"(≤{self.anchor_reproj_threshold:.1f}), "
                            f"mean_sil={anchor_metrics['mean_sil']:.4f} "
                            f"(≤{self.anchor_sil_threshold:.4f}), "
                            f"mean_out={anchor_metrics['mean_outside']:.4f} "
                            f"(≤{self.anchor_outside_threshold:.4f})"
                        )
                        if self.delay_refinement_until_anchor:
                            cfg=self._delay_refinement_cfg(cfg)

                if anchor is None and self.delay_refinement_until_anchor:
                    cfg=self._delay_refinement_cfg(cfg)

                opt=self._make_optimizer(betas, cos, bp, go, tr, lfs, cfg)

            opt.zero_grad()
            total=0.0
            ll={}
            cof=self._chest_offsets_full(cos)
            l2,sm,ow,iwd=self._chest_regularization(cos)
            lreg=self.chest_offset_l2_weight*l2 + self.chest_smooth_weight*sm + self.chest_outward_prior_weight*ow + self.chest_inward_prior_weight*iwd

            for i in range(num_images):
                fl=self._current_focal(lfs, i)
                out=self.model(betas=betas, body_pose=bp[i], global_orient=go[i], transl=None, return_verts=True)
                verts=out.vertices + cof
                joints=SMPLXJointMapper.smplx_to_coco17(out.joints)
                pj=self._project_points_screen(joints, f=fl, cc=cc, t=tr[i])

                rend=self.renderer.render(vertices=verts, faces=self.model.faces_tensor.unsqueeze(0), focal_length=fl, principal_point=cc, translation=tr[i])
                rm=rend[...,3].unsqueeze(1)
                cr=self.renderer.render(vertices=verts, faces=self.chest_faces, focal_length=fl, principal_point=cc, translation=tr[i])
                cm=cr[...,3].unsqueeze(1)

                md=meta[i]
                vm=torch.ones_like(masks[i])
                border=int(self.image_size*0.12)
                if md.get("truncated_top", False): vm[:,:,:border,:]*=0.4
                if md.get("truncated_bottom", False): vm[:,:,-border:,:]*=0.4
                if md.get("truncated_left", False): vm[:,:,:,:border]*=0.4
                if md.get("truncated_right", False): vm[:,:,:,-border:]*=0.4

                core_prior, arm_prior, torso_prior, pelvis_prior, thigh_prior = self._pose_region_priors(gtk[i], conf[i], masks[i])
                arm_reduce=float(cfg.get("arm_reduce", 0.0))
                silhouette_valid=(valids[i]*(1.0-arm_reduce*arm_prior)).clamp(min=0.20, max=1.0)

                # Explicit target-mask-gated core target:
                #   pose_core_target = pose_core_prior * target_mask * validity
                # This prevents the synthetic pose core from demanding coverage
                # where the real target silhouette or validity mask says there is
                # no reliable body evidence.
                pose_core_target=(core_prior*masks[i]*silhouette_valid).clamp(0.0,1.0)
                pose_core_weight=(core_prior*silhouette_valid).clamp(0.0,1.0)

                torso_target=(torso_prior*masks[i]*silhouette_valid).clamp(0.0,1.0)
                torso_weight=(torso_prior*silhouette_valid).clamp(0.0,1.0)

                pelvis_target=(pelvis_prior*masks[i]*silhouette_valid).clamp(0.0,1.0)
                pelvis_weight=(pelvis_prior*silhouette_valid).clamp(0.0,1.0)

                thigh_target=(thigh_prior*masks[i]*silhouette_valid).clamp(0.0,1.0)
                thigh_weight=(thigh_prior*silhouette_valid).clamp(0.0,1.0)

                vm=vm*silhouette_valid

                rw=BodyRegionWeights.create_weight_map(self.image_size, self.image_size, self.device)
                if rw.dim()==3: rw=rw.unsqueeze(0)
                cv=float(md.get("chest_visible",0.0)); hv=float(md.get("hip_visible",0.0))
                rw=rw*self.sil_region_weights*(1.0+0.20*cv*self.region_maps["breast"])*(1.0+0.12*hv*self.region_maps["glutes"])

                pose_kpw=torch.tensor(pose_keypoint_weight_all[i], dtype=torch.float32, device=self.device)
                if cfg.get("torso_only", False):
                    lkp=self._torso_keypoint_loss(pj, gtk[i], conf[i])*pose_kpw
                    lbone=self._torso_bone_loss(pj, gtk[i], conf[i])*pose_kpw
                    lctr=self._torso_center_scale_loss(pj, gtk[i], conf[i])*pose_kpw
                else:
                    lkp=self._weighted_keypoint_loss(pj, gtk[i], conf[i])*pose_kpw
                    lbone=self._bone_direction_loss(pj, gtk[i], conf[i])*pose_kpw
                    lctr=self._keypoint_center_scale_loss(pj, gtk[i], conf[i])*pose_kpw
                lreproj=self._pose_reprojection_pixel_error(pj, gtk[i], conf[i])
                pose_gate = 1.0 if float(pose_quality_all[i]) >= self.lower_body_pose_gate_threshold else max(0.25, float(pose_quality_all[i]) / max(self.lower_body_pose_gate_threshold, 1e-6))
                llower, lower_reproj_px, pelvis_ctr_l, thigh_dir_l = self._lower_body_keypoint_loss(
                    pj, gtk[i], conf[i], image_weight=pose_kpw * pose_gate
                )

                lsil=silhouette_loss(rm, masks[i], vm, rw)
                ldist=self._distance_silhouette_loss(rm, masks[i], do_all[i], di_all[i], silhouette_valid)
                liou=self._iou_loss(rm, masks[i], silhouette_valid)
                ledge=self._edge_loss(rm, masks[i], silhouette_valid)
                lmask_stats=self._mask_stats_loss(rm, masks[i], silhouette_valid)
                ltight, lfp, lfn, loutside = self._tight_fit_losses(rm, masks[i], silhouette_valid)
                lcore_fp, lcore_fn, lcore_cov = self._core_tight_fit_losses(rm, pose_core_target, pose_core_weight)
                _, ltorso_fn, _ = self._core_tight_fit_losses(rm, torso_target, torso_weight)
                _, lpelvis_fn, _ = self._core_tight_fit_losses(rm, pelvis_target, pelvis_weight)
                _, lthigh_fn, _ = self._core_tight_fit_losses(rm, thigh_target, thigh_weight)
                lwidth=self._width_loss(rm, tw_all[i], md)
                larea=self._regional_area_loss(rm, ta_all[i], md)
                lbloat=self._anti_bloat_loss(rm, masks[i], silhouette_valid)

                lcproj,lcw,lca=self._projected_chest_loss(cm, masks[i], tw_all[i], md)
                lbw,lba,lbc=self._bilateral_breast_loss(cm, masks[i], pj, gtk[i], md)
                lcleav=self._cleavage_bridge_loss(cm, masks[i], pj, md)
                lgap=self._interbreast_gap_loss(rm, cm, masks[i], pj, md)
                lstern=self._sternum_flatten_loss(cos)

                labd=self._abdomen_guard_loss(rm, masks[i], md)
                labd_area=self._abdomen_area_loss(rm, ta_all[i], md)
                lwaist=self._waist_guard_loss(rm, tw_all[i], md)
                lgw,lga,lgb=self._glute_shape_loss(rm, tw_all[i], ta_all[i], md)

                lshape=shape_prior_loss(betas)
                lbeta=((betas-initial_betas)**2).mean()
                lpose=pose_prior_loss(bp[i])
                ltrans=translation_loss(tr[i])
                lfocal=self._focal_regularization(lfs)

                ljoint_anchor,lpose_anchor,lcamera_anchor,lbeta_anchor,lsil_guard = self._anchor_losses(
                    pj=pj,
                    rm=rm,
                    betas=betas,
                    bp_i=bp[i],
                    go_i=go[i],
                    tr_i=tr[i],
                    lfs=lfs,
                    anchor=anchor,
                    image_index=i,
                    sil_loss=lsil,
                    outside_loss=loutside,
                    iou_loss_value=liou,
                    valid=silhouette_valid,
                )
                if anchor is not None:
                    lchest_resid=self._chest_residual_loss(cm, masks[i], anchor["rendered_masks"][i], silhouette_valid)
                else:
                    lchest_resid=torch.tensor(0.0, device=self.device)

                il=iw[i]*(
                    cfg["kp"]*lkp +
                    cfg["bone"]*lbone +
                    cfg["center"]*lctr +
                    cfg.get("lb_kp",0.0)*llower +
                    cfg.get("lb_reproj",0.0)*(lower_reproj_px/float(self.image_size)) +
                    cfg.get("lb_ctr",0.0)*pelvis_ctr_l +
                    cfg.get("lb_dir",0.0)*thigh_dir_l +
                    cfg["sil"]*lsil +
                    cfg["dist"]*ldist +
                    cfg["iou"]*liou +
                    cfg["edge"]*ledge +
                    cfg["mask_stats"]*lmask_stats +
                    cfg["tight"]*ltight +
                    cfg["outside"]*loutside +
                    cfg["core_fp"]*lcore_fp +
                    cfg["core_fn"]*lcore_fn +
                    cfg["core_cov"]*lcore_cov +
                    cfg["width"]*lwidth +
                    cfg["area"]*larea +
                    cfg["bloat"]*lbloat +
                    cfg["chest_proj"]*lcproj +
                    cfg["chest_width"]*lcw +
                    cfg["chest_area"]*lca +
                    cfg["bilat_w"]*lbw +
                    cfg["bilat_a"]*lba +
                    cfg["bilat_c"]*lbc +
                    cfg["cleavage"]*lcleav +
                    cfg["gap"]*lgap +
                    cfg["sternum"]*lstern +
                    cfg["abdomen"]*labd +
                    cfg["abdomen_area"]*labd_area +
                    cfg["waist"]*lwaist +
                    cfg["glute_w"]*lgw +
                    cfg["glute_a"]*lga +
                    cfg["glute_bloat"]*lgb +
                    cfg["shape"]*lshape +
                    cfg["beta"]*lbeta +
                    cfg["pose"]*lpose +
                    cfg["trans"]*ltrans +
                    cfg["focal_reg"]*lfocal +
                    cfg.get("anchor_joint",0.0)*ljoint_anchor +
                    cfg.get("anchor_pose",0.0)*lpose_anchor +
                    cfg.get("anchor_camera",0.0)*lcamera_anchor +
                    cfg.get("anchor_beta",0.0)*lbeta_anchor +
                    cfg.get("sil_guard",0.0)*lsil_guard +
                    cfg.get("chest_residual",0.0)*lchest_resid
                )

                total += il
                ll={"kp":lkp,"bone":lbone,"center":lctr,"reproj":lreproj,"sil":lsil,"fp":lfp,"fn":lfn,"outside":loutside,"tight":ltight,"core_fp":lcore_fp,"core_fn":lcore_fn,"core_cov":lcore_cov,"torso_fn":ltorso_fn,"pelvis_fn":lpelvis_fn,"thigh_fn":lthigh_fn,
                    "lower_reproj_px":lower_reproj_px,"pelvis_ctr":pelvis_ctr_l,"thigh_dir":thigh_dir_l,
                    "anchor_joint":ljoint_anchor,"anchor_pose":lpose_anchor,"anchor_camera":lcamera_anchor,"anchor_beta":lbeta_anchor,"sil_guard":lsil_guard,"chest_residual":lchest_resid,
                    "dist":ldist,"iou":liou,"edge":ledge,"mask_stats":lmask_stats,
                    "width":lwidth,"area":larea,"bloat":lbloat,
                    "chest_proj":lcproj,"chest_width":lcw,"chest_area":lca,
                    "bilat_w":lbw,"bilat_a":lba,"bilat_c":lbc,"cleavage":lcleav,"gap":lgap,"sternum":lstern,
                    "abdomen":labd,"abdomen_area":labd_area,"waist":lwaist,
                    "glute_w":lgw,"glute_a":lga,"glute_bloat":lgb,
                    "chest_reg":lreg,"chest_l2":l2,"chest_smooth":sm,"shape":lshape,"beta":lbeta,"pose":lpose,"trans":ltrans,"focal":fl[0,0],"valid":valids[i].mean(),"pose_q":torch.tensor(pose_quality_all[i],device=self.device),"pose_kw":torch.tensor(pose_keypoint_weight_all[i],device=self.device),"pose_iw":torch.tensor(pose_image_weight_all[i],device=self.device)}

                if self.debug and (it % self.debug_every == 0 or it == iterations - 1) and i < self.debug_max_images:
                    self._save_render_debug(
                        debug_dir=debug_dir,
                        iteration=it,
                        image_index=i,
                        target_mask=masks[i],
                        rendered_mask=rm,
                        chest_mask=cm,
                        target_joints=gtk[i],
                        projected_joints=pj,
                        confidence=conf[i],
                        validity=valids[i],
                        core_prior=pose_core_target,
                        arm_prior=arm_prior,
                    )

            total = total + cfg["chest_reg"]*lreg
            total.backward()
            opt.step()

            with torch.no_grad():
                betas.clamp_(-2.5, 2.5)
                if cos is not None:
                    cos.clamp_(-self.chest_offset_limit, self.chest_offset_limit)
                    smc=self._smooth_chest_scalars(cos, steps=max(1, self.chest_smooth_steps//2), alpha=self.chest_smooth_alpha)
                    cos.copy_(torch.clamp(smc, -self.chest_offset_limit, self.chest_offset_limit))
                for t in tr: t[:,2].clamp_(2.0, 10.0)
                if lfs is not None:
                    lfs.clamp_(np.log(800.0/self.base_focal), np.log(4000.0/self.base_focal))

            if it % 10 == 0:
                prog.set_postfix({
                    "phase":cfg["name"],
                    "total":f"{total.item():.2f}",
                    "iou":f"{ll['iou'].item():.4f}",
                    "sil":f"{ll['sil'].item():.4f}",
                    "out":f"{ll['outside'].item():.4f}",
                    "fp":f"{ll['fp'].item():.4f}",
                    "fn":f"{ll['fn'].item():.4f}",
                    "abd":f"{ll['abdomen'].item():.4f}",
                    "gl":f"{ll['glute_a'].item():.4f}",
                    "ch":f"{ll['chest_proj'].item():.4f}",
                    "pq":f"{ll['pose_q'].item():.2f}",
                    "rpx":f"{ll['reproj'].item():.1f}",
                    "lrpx":f"{ll['lower_reproj_px'].item():.1f}",
                    "valid":f"{ll['valid'].item():.2f}",
                })

            if it % 25 == 0:
                tqdm.write(
                    f"\n[iter {it:04d}] phase={cfg['name']}\n"
                    f"  total:       {total.item():.6f}\n"
                    f"  kp:          {ll['kp'].item():.6f}\n"
                    f"  bone:        {ll['bone'].item():.6f}\n"
                    f"  ctr:         {ll['center'].item():.6f}\n"
                    f"  reproj_px:   {ll['reproj'].item():.2f}\n"
                    f"  mask_stats:  {ll['mask_stats'].item():.6f}\n"
                    f"  valid:       {ll['valid'].item():.6f}\n"
                    f"  sil:         {ll['sil'].item():.6f}\n"
                    f"  fp:          {ll['fp'].item():.6f}\n"
                    f"  fn:          {ll['fn'].item():.6f}\n"
                    f"  outside:     {ll['outside'].item():.6f}\n"
                    f"  tight:       {ll['tight'].item():.6f}\n"
                    f"  core_fp:     {ll['core_fp'].item():.6f}\n"
                    f"  core_fn:     {ll['core_fn'].item():.6f}\n"
                    f"  torso_fn:    {ll['torso_fn'].item():.6f}\n"
                    f"  pelvis_fn:   {ll['pelvis_fn'].item():.6f}\n"
                    f"  thigh_fn:    {ll['thigh_fn'].item():.6f}\n"
                    f"  lower_rpx:   {ll['lower_reproj_px'].item():.2f}\n"
                    f"  pelvis_ctr:  {ll['pelvis_ctr'].item():.6f}\n"
                    f"  thigh_dir:   {ll['thigh_dir'].item():.6f}\n"
                    f"  dist:        {ll['dist'].item():.6f}\n"
                    f"  iou:         {ll['iou'].item():.6f}\n"
                    f"  edge:        {ll['edge'].item():.6f}\n"
                    f"  width:       {ll['width'].item():.6f}\n"
                    f"  area:        {ll['area'].item():.6f}\n"
                    f"  bloat:       {ll['bloat'].item():.6f}\n"
                    f"  chest_proj:  {ll['chest_proj'].item():.6f}\n"
                    f"  chest_width: {ll['chest_width'].item():.6f}\n"
                    f"  chest_area:  {ll['chest_area'].item():.6f}\n"
                    f"  bilat_width: {ll['bilat_w'].item():.6f}\n"
                    f"  bilat_area:  {ll['bilat_a'].item():.6f}\n"
                    f"  bilat_ctr:   {ll['bilat_c'].item():.6f}\n"
                    f"  cleavage:    {ll['cleavage'].item():.6f}\n"
                    f"  gap:         {ll['gap'].item():.6f}\n"
                    f"  sternum:     {ll['sternum'].item():.6f}\n"
                    f"  abdomen:     {ll['abdomen'].item():.6f}\n"
                    f"  abd_area:    {ll['abdomen_area'].item():.6f}\n"
                    f"  waist:       {ll['waist'].item():.6f}\n"
                    f"  glute_width: {ll['glute_w'].item():.6f}\n"
                    f"  glute_area:  {ll['glute_a'].item():.6f}\n"
                    f"  glute_bloat: {ll['glute_bloat'].item():.6f}\n"
                    f"  chest_reg:   {ll['chest_reg'].item():.8f}\n"
                    f"  chest_l2:    {ll['chest_l2'].item():.8f}\n"
                    f"  chest_smooth:{ll['chest_smooth'].item():.8f}\n"
                    f"  shape:       {ll['shape'].item():.6f}\n"
                    f"  beta:        {ll['beta'].item():.6f}\n"
                    f"  pose:        {ll['pose'].item():.6f}\n"
                    f"  trans:       {ll['trans'].item():.6f}\n"
                    f"  focal:       {float(ll['focal'].item() if torch.is_tensor(ll['focal']) else ll['focal']):.2f}\n"
                    f"  anchor_joint:{ll['anchor_joint'].item():.6f}\n"
                    f"  anchor_pose: {ll['anchor_pose'].item():.6f}\n"
                    f"  anchor_cam:  {ll['anchor_camera'].item():.6f}\n"
                    f"  anchor_beta: {ll['anchor_beta'].item():.6f}\n"
                    f"  sil_guard:   {ll['sil_guard'].item():.6f}\n"
                    f"  chest_resid: {ll['chest_residual'].item():.6f}\n"
                    f"  pose_q:      {ll['pose_q'].item():.3f}\n"
                    f"  pose_kw:     {ll['pose_kw'].item():.3f}\n"
                    f"  pose_iw:     {ll['pose_iw'].item():.3f}"
                )

        with torch.no_grad():
            ff=self._current_focal(lfs)
            fco=self._chest_offsets_full(cos)
            nout=self.model(
                betas=betas,
                body_pose=torch.zeros(1,63,device=self.device),
                global_orient=torch.zeros(1,3,device=self.device),
                transl=None,
                return_verts=True,
            )
            nverts=nout.vertices + fco
            piv=[]; pij=[]
            for i in range(num_images):
                oi=self.model(betas=betas, body_pose=bp[i], global_orient=go[i], transl=None, return_verts=True)
                piv.append((oi.vertices+fco).detach().cpu().numpy()[0])
                pij.append(oi.joints.detach().cpu().numpy()[0])
            vrm=RegionAwareMasks.template_vertex_masks(nverts.detach()[0])

        res={
            "betas":betas.detach().cpu().numpy(),
            "vertices":nverts.detach().cpu().numpy(),
            "joints":nout.joints.detach().cpu().numpy(),
            "faces":self.model.faces,
            "num_images":num_images,
            "per_image_vertices":np.stack(piv, axis=0),
            "per_image_joints":np.stack(pij, axis=0),
            "body_poses":np.stack([p.detach().cpu().numpy()[0] for p in bp], axis=0),
            "global_orients":np.stack([g.detach().cpu().numpy()[0] for g in go], axis=0),
            "translations":np.stack([t.detach().cpu().numpy()[0] for t in tr], axis=0),
            "focal_length":ff.detach().cpu().numpy(),
            "camera_center":cc.detach().cpu().numpy(),
            "chest_vertex_ids":self.chest_vertex_ids.detach().cpu().numpy(),
            "chest_offsets":np.zeros((1,0,1), dtype=np.float32) if cos is None else cos.detach().cpu().numpy(),
            "chest_offset_limit":np.array([self.chest_offset_limit], dtype=np.float32),
            "has_sternum_cleavage_losses":np.array([1], dtype=np.int32),
            "has_abdomen_glute_guards":np.array([1], dtype=np.int32),
            "has_interbreast_gap_loss":np.array([1], dtype=np.int32),
            "has_render_debug_export":np.array([1], dtype=np.int32),
            "has_pose_quality_weighting":np.array([1], dtype=np.int32),
            "has_pose_lock_schedule":np.array([1], dtype=np.int32),
            "has_mask_stats_loss":np.array([1], dtype=np.int32),
            "has_keypoint_debug_overlay":np.array([1], dtype=np.int32),
            "has_torso_center_scale_fix":np.array([1], dtype=np.int32),
            "pose_quality_scores":np.array(pose_quality_all, dtype=np.float32),
            "pose_image_weights":np.array(pose_image_weight_all, dtype=np.float32),
            "pose_keypoint_weights":np.array(pose_keypoint_weight_all, dtype=np.float32),
            "silhouette_validity_means":np.array(validity_mean_all, dtype=np.float32),
            "used_hair_aware_masks":np.array([1 if self.use_hair_aware_masks else 0], dtype=np.int32),
            "hair_unknown_weight":np.array([self.hair_unknown_weight], dtype=np.float32),
            "has_hair_aware_silhouette_validity":np.array([1], dtype=np.int32),
            "has_adaptive_tight_fit":np.array([1], dtype=np.int32),
            "has_body_core_tight_fit":np.array([1], dtype=np.int32),
            "has_pose_aware_core_masks":np.array([1], dtype=np.int32),
            "has_tight_fit_losses_fix":np.array([1], dtype=np.int32),
            "arm_silhouette_reduction":np.array([self.arm_silhouette_reduction], dtype=np.float32),
            "core_coverage_adaptive_max":np.array([self.core_coverage_adaptive_max], dtype=np.float32),
            "pose_core_dilate_px":np.array([self.pose_core_dilate_px], dtype=np.int32),
            "tight_fit_weight":np.array([self.tight_fit_weight], dtype=np.float32),
            "outside_target_weight":np.array([self.outside_target_weight], dtype=np.float32),
            "core_coverage_weight":np.array([self.core_coverage_weight], dtype=np.float32),
            "anchor_reproj_threshold":np.array([self.anchor_reproj_threshold], dtype=np.float32),
            "anchor_sil_threshold":np.array([self.anchor_sil_threshold], dtype=np.float32),
            "anchor_outside_threshold":np.array([self.anchor_outside_threshold], dtype=np.float32),
            "per_image_focal_log_scales":np.array([] if lfs is None else lfs.detach().cpu().numpy(), dtype=np.float32),
            "pose_backends":np.array(pose_backend_all),
            "pose_models":np.array(pose_model_all),
        }
        for name,mask in RegionAwareMasks.masks_to_numpy(vrm).items():
            res[f"vertex_mask_{name}"]=mask
        cm=np.zeros((self.template_normals.shape[1],), dtype=bool)
        cm[self.chest_vertex_ids.detach().cpu().numpy()]=True
        res["vertex_mask_chest_optimizer"]=cm

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # EXPORT PER-IMAGE CAMERA JSON
        # =====================================================
        camera_dir = output_path.parent / "camera"
        camera_dir.mkdir(parents=True, exist_ok=True)

        cameras_index = []
        focal_all_np = ff.detach().cpu().numpy()
        center_np = cc.detach().cpu().numpy()[0].tolist()

        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)
            stem = image_path.stem
            camera_json_path = camera_dir / f"{stem}_camera.json"

            camera_data = {
                "image_path": str(image_path),
                "image_stem": stem,
                "image_size": int(self.image_size),
                "focal_length": [
                    float(focal_all_np[i,0] if focal_all_np.ndim==2 else focal_all_np[0]),
                    float(focal_all_np[i,1] if focal_all_np.ndim==2 else focal_all_np[0]),
                ],
                "camera_center": [
                    float(center_np[0]),
                    float(center_np[1]),
                ],
                "translation": tr[i].detach().cpu().numpy()[0].astype(float).tolist(),
                "global_orient": go[i].detach().cpu().numpy()[0].astype(float).tolist(),
                "body_pose": bp[i].detach().cpu().numpy()[0].astype(float).tolist(),
                "camera_model": "pytorch3d_perspective_screen",
                "in_ndc": False,
                "pose_quality_score": float(pose_quality_all[i]),
                "pose_image_weight": float(pose_image_weight_all[i]),
                "pose_keypoint_weight": float(pose_keypoint_weight_all[i]),
                "silhouette_validity_mean": float(validity_mean_all[i]),
                "hair_aware_masks_enabled": bool(self.use_hair_aware_masks),
                "pose_backend": str(pose_backend_all[i]),
                "pose_model": str(pose_model_all[i]),
            }

            with open(camera_json_path, "w") as f:
                json.dump(camera_data, f, indent=2)

            cameras_index.append({
                "image_path": str(image_path),
                "image_stem": stem,
                "camera_json": str(camera_json_path),
            })

        cameras_index_path = camera_dir / "cameras_index.json"

        with open(cameras_index_path, "w") as f:
            json.dump(cameras_index, f, indent=2)

        res["camera_dir"] = np.array([str(camera_dir)])
        res["cameras_index_path"] = np.array([str(cameras_index_path)])

        np.savez(output_path, **res)

        print(f"✅ Camera JSON exported: {camera_dir}")
        print(f"\n✅ Saved canonical body:\n{output_path}")
        return res
