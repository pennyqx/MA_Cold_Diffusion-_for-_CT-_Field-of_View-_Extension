"""
=============================================================================
CT数据预处理主管道 (Main Preprocessing Pipeline)
=============================================================================

本模块是数据预处理的统一入口，支持:
- CT-ORG (NIfTI格式)
- LIDC-IDRI (DICOM格式)

主要功能:
1. 加载CT体积数据
2. 重采样到统一分辨率
3. 裁剪/填充到目标尺寸
4. 前向投影生成Sinogram
5. 切片提取和归一化

=============================================================================
维度约定 (Dimension Conventions) - 非常重要!
=============================================================================

【CT体积数据】
- 标准医学影像: (Z, Y, X) = (轴向切片数, 高度/行, 宽度/列)
- Z: 轴向方向 (从头到脚或从脚到头)
- Y: 冠状方向 (前后)
- X: 矢状方向 (左右)

【NIfTI文件】
- nibabel加载后默认: (X, Y, Z) - RAS方向
- 需要转置为: (Z, Y, X) 以符合医学影像惯例

【DICOM文件】
- 单张切片: (rows, columns) = (Y, X)
- 堆叠后: (Z, Y, X)

【PyRoNN投影】
- PyRoNN内部detector_shape: (height, width)
- 我们的约定detector_shape: (width, height)
- swap_detector_axis=True 确保一致性

【输出Sinogram】
- 我们的约定: (n_projections, detector_width, detector_height)
- 即: (360, 640, 560) 表示360个角度，探测器640宽×560高

【2D切片提取】
- 从3D Sinogram提取: sinogram_2d = sinogram_3d[:, :, z_idx]
- 2D形状: (n_projections, detector_width) = (360, 640)
=============================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import warnings
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VolumeConfig:
    """CT体积参数配置"""
    # 目标体积形状 (Z, Y, X)
    target_shape: Tuple[int, int, int] = (128, 512, 512)

    # 目标体素间距 (sz, sy, sx) in mm
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0)

    # HU值裁剪范围
    hu_min: float = -1000.0  # 空气
    hu_max: float = 1000.0   # 骨骼以下

    # 有效切片厚度范围 (mm)
    valid_slice_thickness_range: Tuple[float, float] = (0.5, 3.0)

    # 最小有效切片数
    min_valid_slices: int = 64


@dataclass
class DetectorConfig:
    """探测器参数配置"""
    # 探测器尺寸 (width, height) - 我们的约定
    detector_width: int = 640
    detector_height: int = 560

    # 探测器像素间距 (mm)
    detector_spacing_x: float = 0.5  # width方向
    detector_spacing_y: float = 0.5  # height方向

    # 投影数目
    n_projections: int = 360

    # 角度范围 (rad)
    angular_range: float = 2 * np.pi

    # 源-探测器距离 (mm)
    source_detector_distance: float = 1200.0

    # 源-等中心距离 (mm)
    source_isocenter_distance: float = 750.0


@dataclass
class PreprocessingResult:
    """预处理结果"""
    success: bool = False
    sinogram_path: Optional[str] = None
    volume_path: Optional[str] = None
    original_shape: Optional[Tuple[int, int, int]] = None
    original_spacing: Optional[Tuple[float, float, float]] = None
    final_shape: Optional[Tuple[int, int, int]] = None
    sinogram_shape: Optional[Tuple[int, int, int]] = None
    patient_id: str = ""
    error_message: str = ""
    metadata: Dict = field(default_factory=dict)


class VolumeLoader(ABC):
    """体积数据加载器基类"""

    @abstractmethod
    def load(self, path: str) -> Tuple[np.ndarray, Dict]:
        """
        加载CT体积数据

        Args:
            path: 文件或目录路径

        Returns:
            (volume, metadata)
            - volume: shape=(Z, Y, X), dtype=float32, 单位HU
            - metadata: 包含spacing, patient_id等信息
        """
        pass


class NIfTILoader(VolumeLoader):
    """NIfTI格式加载器 (用于CT-ORG数据集)"""

    def load(self, nifti_path: str) -> Tuple[np.ndarray, Dict]:
        """
        加载NIfTI文件

        维度变换说明:
        1. nibabel加载: (X, Y, Z) - RAS坐标系
        2. 转置为: (Z, Y, X) - 标准医学影像坐标系
        3. spacing也相应调整: (sx, sy, sz) -> (sz, sy, sx)
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("请安装nibabel: pip install nibabel")

        logger.info(f"加载NIfTI文件: {nifti_path}")

        img = nib.load(nifti_path)

        # 获取原始数据 - nibabel默认 (X, Y, Z)
        data_xyz = np.asarray(img.dataobj, dtype=np.float32)

        # 从affine矩阵获取体素间距
        affine = img.affine
        spacing_xyz = tuple(np.abs(np.diag(affine)[:3]))

        logger.debug(f"  原始形状 (X,Y,Z): {data_xyz.shape}")
        logger.debug(f"  原始间距 (sx,sy,sz): {spacing_xyz}")

        # 转置为 (Z, Y, X)
        volume_zyx = np.transpose(data_xyz, (2, 1, 0))
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

        logger.info(f"  转换后形状 (Z,Y,X): {volume_zyx.shape}")
        logger.info(f"  转换后间距 (sz,sy,sx): {spacing_zyx}")

        # 提取patient_id
        path_obj = Path(nifti_path)
        patient_id = path_obj.stem.replace('.nii', '')

        metadata = {
            'spacing': spacing_zyx,  # (sz, sy, sx)
            'original_shape': volume_zyx.shape,
            'patient_id': patient_id,
            'source': 'nifti',
            'affine': affine
        }

        return volume_zyx, metadata


class DICOMLoader(VolumeLoader):
    """DICOM格式加载器 (用于LIDC-IDRI数据集)"""

    def __init__(self, require_ct: bool = True):
        """
        Args:
            require_ct: 是否只加载CT图像 (排除其他模态)
        """
        self.require_ct = require_ct

    @staticmethod
    def _is_archive(path_str: str) -> bool:
        lower = path_str.lower()
        return lower.endswith('.zip') or lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz')

    @staticmethod
    def _split_archive_path(path_str: str) -> Tuple[str, Optional[str]]:
        # "archive_path::inner_prefix"
        if '::' in path_str:
            archive_path, inner_prefix = path_str.split('::', 1)
            inner_prefix = inner_prefix.strip('/')
            return archive_path, inner_prefix or None
        return path_str, None

    @staticmethod
    def _is_dicom_name(name: str) -> bool:
        base = os.path.basename(name)
        return base.lower().endswith('.dcm') or '.' not in base

    def _list_archive_members(self, archive_path: str, inner_prefix: Optional[str]) -> List[str]:
        members: List[str] = []
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    if inner_prefix and not name.startswith(inner_prefix + '/'):
                        continue
                    if self._is_dicom_name(name):
                        members.append(name)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tf:
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    name = info.name
                    if inner_prefix and not name.startswith(inner_prefix + '/'):
                        continue
                    if self._is_dicom_name(name):
                        members.append(name)
        # Stable order to make behavior deterministic
        members.sort()
        return members

    def _iter_dicom_from_archive(self, archive_path: str, members: List[str]):
        import io
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in members:
                    with zf.open(name, 'r') as f:
                        data = f.read()
                    yield name, io.BytesIO(data)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tf:
                for name in members:
                    f = tf.extractfile(name)
                    if f is None:
                        continue
                    data = f.read()
                    f.close()
                    yield name, io.BytesIO(data)

    def load(self, dicom_dir: str) -> Tuple[np.ndarray, Dict]:
        """
        加载DICOM序列

        维度说明:
        1. 单张DICOM: pixel_array shape = (rows, cols) = (Y, X)
        2. 按位置排序后堆叠: (n_slices, rows, cols) = (Z, Y, X)
        3. 无需转置，直接符合医学影像约定
        """
        try:
            import pydicom
            from pydicom.filereader import dcmread
        except ImportError:
            raise ImportError("请安装pydicom: pip install pydicom")

        archive_path = None
        inner_prefix = None
        if self._is_archive(dicom_dir) or '::' in dicom_dir:
            archive_path, inner_prefix = self._split_archive_path(dicom_dir)

        if archive_path and self._is_archive(archive_path):
            logger.info(f"扫描DICOM压缩包: {archive_path}" + (f"::{inner_prefix}" if inner_prefix else ""))
            dicom_files = self._list_archive_members(archive_path, inner_prefix)
            if not dicom_files:
                raise ValueError(f"未找到DICOM文件: {dicom_dir}")
            logger.info(f"  找到 {len(dicom_files)} 个DICOM文件")
        else:
            logger.info(f"扫描DICOM目录: {dicom_dir}")
            # 收集所有DICOM文件
            dicom_files = []
            for root, _, files in os.walk(dicom_dir):
                for f in files:
                    if f.lower().endswith('.dcm') or '.' not in f:
                        dicom_files.append(os.path.join(root, f))

            if not dicom_files:
                raise ValueError(f"未找到DICOM文件: {dicom_dir}")

            logger.info(f"  找到 {len(dicom_files)} 个DICOM文件")

        # 加载并过滤
        slices = []
        if archive_path and self._is_archive(archive_path):
            for name, bio in self._iter_dicom_from_archive(archive_path, dicom_files):
                try:
                    ds = dcmread(bio, force=True)

                    # 检查是否为CT
                    if self.require_ct and hasattr(ds, 'Modality'):
                        if ds.Modality != 'CT':
                            continue

                    # 检查是否有像素数据
                    if hasattr(ds, 'pixel_array'):
                        slices.append(ds)
                except Exception as e:
                    logger.warning(f"  跳过文件 {name}: {e}")
                    continue
        else:
            for f in dicom_files:
                try:
                    ds = dcmread(f, force=True)

                    # 检查是否为CT
                    if self.require_ct and hasattr(ds, 'Modality'):
                        if ds.Modality != 'CT':
                            continue

                    # 检查是否有像素数据
                    if hasattr(ds, 'pixel_array'):
                        slices.append(ds)
                except Exception as e:
                    logger.warning(f"  跳过文件 {f}: {e}")
                    continue

        if len(slices) < 10:
            raise ValueError(f"有效DICOM切片不足: {len(slices)}")

        logger.info(f"  有效CT切片: {len(slices)}")

        # 按ImagePositionPatient[2] (Z坐标) 排序
        def get_z_position(ds):
            if hasattr(ds, 'ImagePositionPatient'):
                return float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            elif hasattr(ds, 'InstanceNumber'):
                return float(ds.InstanceNumber)
            return 0.0

        slices.sort(key=get_z_position)

        # 获取元数据
        first = slices[0]

        # 像素间距 (dy, dx) in mm
        if hasattr(first, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in first.PixelSpacing]
        else:
            pixel_spacing = [1.0, 1.0]

        # 切片厚度
        if hasattr(first, 'SliceThickness'):
            slice_thickness = float(first.SliceThickness)
        else:
            # 从相邻切片位置计算
            if len(slices) > 1:
                z_positions = [get_z_position(s) for s in slices[:10]]
                slice_thickness = np.median(np.diff(z_positions))
            else:
                slice_thickness = 1.0

        # 堆叠像素数据
        # pixel_array shape = (rows, cols) = (Y, X)
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
        # 堆叠后: (Z, Y, X)

        # 转换为HU值
        slope = float(first.RescaleSlope) if hasattr(first, 'RescaleSlope') else 1.0
        intercept = float(first.RescaleIntercept) if hasattr(first, 'RescaleIntercept') else 0.0
        volume = volume * slope + intercept

        logger.info(f"  体积形状 (Z,Y,X): {volume.shape}")
        logger.info(f"  像素间距 (sy,sx): {pixel_spacing}")
        logger.info(f"  切片厚度 (sz): {slice_thickness}")
        logger.info(f"  HU范围: [{volume.min():.1f}, {volume.max():.1f}]")

        # spacing: (sz, sy, sx)
        spacing = (abs(slice_thickness), pixel_spacing[0], pixel_spacing[1])

        # Patient ID
        patient_id = str(first.PatientID) if hasattr(first, 'PatientID') else 'unknown'

        metadata = {
            'spacing': spacing,  # (sz, sy, sx)
            'original_shape': volume.shape,
            'patient_id': patient_id,
            'source': 'dicom',
            'n_slices': len(slices),
            'rescale_slope': slope,
            'rescale_intercept': intercept
        }

        return volume, metadata


class VolumePreprocessor:
    """CT体积预处理器"""

    def __init__(self, config: VolumeConfig):
        self.config = config

    def resample(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        target_spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        重采样到目标间距

        维度约定:
        - volume: (Z, Y, X)
        - spacing: (sz, sy, sx)

        Args:
            volume: 输入体积 (Z, Y, X)
            original_spacing: 原始间距 (sz, sy, sx) mm
            target_spacing: 目标间距 (sz, sy, sx) mm

        Returns:
            重采样后的体积 (Z', Y', X')
        """
        from scipy.ndimage import zoom

        if target_spacing is None:
            target_spacing = self.config.target_spacing

        # 计算缩放因子
        zoom_factors = tuple(o / t for o, t in zip(original_spacing, target_spacing))

        logger.debug(f"  重采样缩放因子 (Z,Y,X): {zoom_factors}")

        resampled = zoom(volume, zoom_factors, order=1)  # 线性插值

        logger.info(f"  重采样: {volume.shape} -> {resampled.shape}")

        return resampled.astype(np.float32)

    def crop_or_pad(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
        fill_value: float = -1000.0  # 空气HU值
    ) -> np.ndarray:
        """
        裁剪或填充到目标尺寸 (居中)

        Args:
            volume: 输入体积 (Z, Y, X)
            target_shape: 目标形状 (Z, Y, X)
            fill_value: 填充值 (默认空气HU)

        Returns:
            处理后的体积
        """
        if target_shape is None:
            target_shape = self.config.target_shape

        current = volume.shape
        target = target_shape

        # 初始化输出
        output = np.full(target, fill_value, dtype=np.float32)

        # 计算每个维度的裁剪/填充
        slices_src = []
        slices_dst = []

        for i, (c, t) in enumerate(zip(current, target)):
            if c >= t:
                # 需要裁剪 - 取中心部分
                start = (c - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(None))
            else:
                # 需要填充 - 居中放置
                start = (t - c) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + c))

        output[tuple(slices_dst)] = volume[tuple(slices_src)]

        logger.info(f"  裁剪/填充: {volume.shape} -> {output.shape}")

        return output

    def clip_hu(
        self,
        volume: np.ndarray,
        hu_min: Optional[float] = None,
        hu_max: Optional[float] = None
    ) -> np.ndarray:
        """裁剪HU值到指定范围"""
        if hu_min is None:
            hu_min = self.config.hu_min
        if hu_max is None:
            hu_max = self.config.hu_max

        clipped = np.clip(volume, hu_min, hu_max)

        logger.debug(f"  HU裁剪: [{hu_min}, {hu_max}]")

        return clipped

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]范围"""
        v_min = self.config.hu_min
        v_max = self.config.hu_max

        normalized = (volume - v_min) / (v_max - v_min)

        return normalized.astype(np.float32)

    def process(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        apply_normalize: bool = False
    ) -> np.ndarray:
        """
        完整预处理流程

        Args:
            volume: 原始体积 (Z, Y, X)
            original_spacing: 原始间距 (sz, sy, sx)
            apply_normalize: 是否归一化到[0,1]

        Returns:
            预处理后的体积
        """
        # 1. 重采样
        volume = self.resample(volume, original_spacing)

        # 2. 裁剪/填充
        volume = self.crop_or_pad(volume)

        # 3. HU裁剪
        volume = self.clip_hu(volume)

        # 4. 可选归一化
        if apply_normalize:
            volume = self.normalize(volume)

        return volume


class SinogramGenerator:
    """Sinogram生成器 (使用PyRoNN)"""

    def __init__(
        self,
        volume_config: VolumeConfig,
        detector_config: DetectorConfig
    ):
        self.vol_cfg = volume_config
        self.det_cfg = detector_config

        self._geometry = None
        self._projector = None

    def _init_pyronn(self):
        """延迟初始化PyRoNN组件"""
        if self._geometry is not None:
            return

        try:
            from pyronn.ct_reconstruction.geometry.geometry import Geometry
            from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
            from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
        except ImportError:
            raise ImportError(
                "PyRoNN未安装。请从 https://github.com/csyben/PYRO-NN 安装"
            )

        logger.info("初始化PyRoNN几何...")

        # 【关键】PyRoNN内部使用 (height, width) 约定
        # 我们使用 swap_detector_axis=True 来适配我们的 (width, height) 约定
        detector_shape_pyronn = (self.det_cfg.detector_height, self.det_cfg.detector_width)
        detector_spacing_pyronn = (self.det_cfg.detector_spacing_y, self.det_cfg.detector_spacing_x)

        logger.debug(f"  PyRoNN detector_shape (H,W): {detector_shape_pyronn}")

        geo = Geometry()
        geo.init_from_parameters(
            volume_shape=self.vol_cfg.target_shape,
            volume_spacing=self.vol_cfg.target_spacing,
            detector_shape=detector_shape_pyronn,
            detector_spacing=detector_spacing_pyronn,
            number_of_projections=self.det_cfg.n_projections,
            angular_range=self.det_cfg.angular_range,
            source_detector_distance=self.det_cfg.source_detector_distance,
            source_isocenter_distance=self.det_cfg.source_isocenter_distance,
            trajectory=circular_trajectory_3d,
            swap_detector_axis=True  # 【关键】确保维度正确
        )

        self._geometry = geo
        self._projector = ConeProjection3D()

        logger.info("  PyRoNN初始化完成")

    def forward_project(self, volume: np.ndarray) -> np.ndarray:
        """
        前向投影: CT体积 -> Sinogram

        维度变换:
        1. 输入 volume: (Z, Y, X)
        2. PyRoNN内部输出: (n_proj, height, width)
        3. 转置为我们的约定: (n_proj, width, height)

        Args:
            volume: CT体积 (Z, Y, X)

        Returns:
            sinogram: (n_projections, detector_width, detector_height)
        """
        import torch

        self._init_pyronn()

        expected_shape = self.vol_cfg.target_shape
        if volume.shape != expected_shape:
            raise ValueError(f"体积形状 {volume.shape} != 预期 {expected_shape}")

        logger.info(f"前向投影: {volume.shape} -> sinogram")

        # 转换为tensor
        volume_tensor = torch.from_numpy(volume[np.newaxis, ...]).float()

        if torch.cuda.is_available():
            volume_tensor = volume_tensor.cuda()

        # 投影
        with torch.no_grad():
            sinogram = self._projector.forward(volume_tensor, **self._geometry)

        # 转回numpy
        if torch.cuda.is_available():
            sinogram = sinogram.cpu()
        sinogram = sinogram.numpy()[0]  # (n_proj, height, width) - PyRoNN输出

        # 【关键】转置为我们的约定: (n_proj, width, height)
        sinogram = np.transpose(sinogram, (0, 2, 1))

        expected_sino_shape = (
            self.det_cfg.n_projections,
            self.det_cfg.detector_width,
            self.det_cfg.detector_height
        )
        assert sinogram.shape == expected_sino_shape, \
            f"Sinogram形状 {sinogram.shape} != 预期 {expected_sino_shape}"

        logger.info(f"  Sinogram形状: {sinogram.shape}")
        logger.info(f"  值范围: [{sinogram.min():.4f}, {sinogram.max():.4f}]")

        return sinogram.astype(np.float32)


def hu_to_mu(volume: np.ndarray, hu_air: float = -1000.0, hu_water: float = 0.0) -> np.ndarray:
    """
    Convert HU values to linear attenuation coefficients (mu).
    This makes the projection input non-negative for physical sinograms.
    """
    mu = (volume - hu_air) / (hu_water - hu_air)
    return np.clip(mu, 0.0, None).astype(np.float32)


class PreprocessingPipeline:
    """
    完整预处理管道

    将CT数据(DICOM/NIfTI)转换为Sinogram
    """

    def __init__(
        self,
        volume_config: Optional[VolumeConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
        save_intermediate: bool = False
    ):
        self.vol_cfg = volume_config or VolumeConfig()
        self.det_cfg = detector_config or DetectorConfig()
        self.save_intermediate = save_intermediate

        # 初始化组件
        self.nifti_loader = NIfTILoader()
        self.dicom_loader = DICOMLoader(require_ct=True)
        self.preprocessor = VolumePreprocessor(self.vol_cfg)
        self.sino_generator = SinogramGenerator(self.vol_cfg, self.det_cfg)

    def process_nifti(
        self,
        nifti_path: str,
        output_dir: str,
        patient_id: Optional[str] = None
    ) -> PreprocessingResult:
        """处理单个NIfTI文件"""
        result = PreprocessingResult()

        try:
            # 加载
            volume, metadata = self.nifti_loader.load(nifti_path)
            result.original_shape = metadata['original_shape']
            result.original_spacing = metadata['spacing']
            result.patient_id = patient_id or metadata.get('patient_id', 'unknown')

            # 验证切片厚度
            sz = metadata['spacing'][0]
            min_t, max_t = self.vol_cfg.valid_slice_thickness_range
            if not (min_t <= sz <= max_t):
                result.error_message = f"切片厚度 {sz:.2f}mm 超出范围 [{min_t}, {max_t}]"
                logger.warning(result.error_message)
                return result

            # 预处理
            volume = self.preprocessor.process(volume, metadata['spacing'])
            result.final_shape = volume.shape

            # 前向投影
            volume = hu_to_mu(volume)
            sinogram = self.sino_generator.forward_project(volume)
            result.sinogram_shape = sinogram.shape

            # 保存
            os.makedirs(output_dir, exist_ok=True)

            sino_filename = f"{result.patient_id}_sinogram_{self.det_cfg.detector_width}x{self.det_cfg.detector_height}.npy"
            sino_path = os.path.join(output_dir, sino_filename)
            np.save(sino_path, sinogram)
            result.sinogram_path = sino_path

            if self.save_intermediate:
                vol_path = os.path.join(output_dir, f"{result.patient_id}_volume.npy")
                np.save(vol_path, volume)
                result.volume_path = vol_path

            result.success = True
            result.metadata = metadata
            logger.info(f"成功处理: {result.patient_id}")

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"处理失败: {e}")

        return result

    def process_dicom(
        self,
        dicom_dir: str,
        output_dir: str,
        patient_id: Optional[str] = None
    ) -> PreprocessingResult:
        """处理DICOM目录"""
        result = PreprocessingResult()

        try:
            # 加载
            volume, metadata = self.dicom_loader.load(dicom_dir)
            result.original_shape = metadata['original_shape']
            result.original_spacing = metadata['spacing']
            result.patient_id = patient_id or metadata.get('patient_id', 'unknown')

            # 验证切片厚度
            sz = metadata['spacing'][0]
            min_t, max_t = self.vol_cfg.valid_slice_thickness_range
            if not (min_t <= sz <= max_t):
                result.error_message = f"切片厚度 {sz:.2f}mm 超出范围 [{min_t}, {max_t}]"
                logger.warning(result.error_message)
                return result

            # 预处理
            volume = self.preprocessor.process(volume, metadata['spacing'])
            result.final_shape = volume.shape

            # 前向投影
            volume = hu_to_mu(volume)
            sinogram = self.sino_generator.forward_project(volume)
            result.sinogram_shape = sinogram.shape

            # 保存
            os.makedirs(output_dir, exist_ok=True)

            sino_filename = f"{result.patient_id}_sinogram_{self.det_cfg.detector_width}x{self.det_cfg.detector_height}.npy"
            sino_path = os.path.join(output_dir, sino_filename)
            np.save(sino_path, sinogram)
            result.sinogram_path = sino_path

            if self.save_intermediate:
                vol_path = os.path.join(output_dir, f"{result.patient_id}_volume.npy")
                np.save(vol_path, volume)
                result.volume_path = vol_path

            result.success = True
            result.metadata = metadata
            logger.info(f"成功处理: {result.patient_id}")

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"处理失败: {e}")

        return result

    def auto_process(
        self,
        input_path: str,
        output_dir: str,
        patient_id: Optional[str] = None
    ) -> PreprocessingResult:
        """自动检测输入类型并处理"""
        path = Path(input_path)

        if path.is_dir():
            # 检查是否包含DICOM文件
            has_dcm = any(path.glob('**/*.dcm')) or any(
                f for f in path.iterdir() if f.is_file() and '.' not in f.name
            )
            if has_dcm:
                return self.process_dicom(input_path, output_dir, patient_id)
            else:
                raise ValueError(f"目录中未找到DICOM文件: {input_path}")
        elif path.suffix in ['.nii', '.gz']:
            return self.process_nifti(input_path, output_dir, patient_id)
        else:
            raise ValueError(f"不支持的输入格式: {input_path}")


def extract_2d_sinograms(
    sinogram_3d_path: str,
    output_dir: str,
    slice_indices: Optional[List[int]] = None,
    slice_step: int = 1
) -> List[str]:
    """
    从3D Sinogram提取2D切片

    维度变换:
    - 输入: (n_proj, width, height) 例如 (360, 640, 560)
    - 输出: (n_proj, width) 例如 (360, 640)

    切片提取方式:
    - sinogram_2d = sinogram_3d[:, :, z_idx]
    - 即沿height维度(第3维)切片

    Args:
        sinogram_3d_path: 3D sinogram路径
        output_dir: 2D切片输出目录
        slice_indices: 指定切片索引 (None表示全部)
        slice_step: 切片步长 (slice_indices为None时使用)

    Returns:
        保存的2D sinogram路径列表
    """
    sinogram_3d = np.load(sinogram_3d_path)

    n_proj, width, height = sinogram_3d.shape
    logger.info(f"3D Sinogram形状: {sinogram_3d.shape}")

    os.makedirs(output_dir, exist_ok=True)

    if slice_indices is None:
        slice_indices = list(range(0, height, slice_step))

    # 提取基础文件名
    base_name = Path(sinogram_3d_path).stem.replace('_sinogram', '')

    saved_paths = []
    for z_idx in slice_indices:
        if z_idx >= height:
            continue

        # 提取2D切片
        sinogram_2d = sinogram_3d[:, :, z_idx]  # (n_proj, width)

        # 保存
        filename = f"{base_name}_slice{z_idx:04d}_sino2d_{width}.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, sinogram_2d)
        saved_paths.append(save_path)

    logger.info(f"提取了 {len(saved_paths)} 个2D切片")

    return saved_paths


def print_dimension_info():
    """打印维度约定信息"""
    info = """
=============================================================================
                        维度约定说明 (Dimension Conventions)
=============================================================================

1. CT体积数据
   - 约定: (Z, Y, X) = (轴向切片数, 高度/行, 宽度/列)
   - 默认形状: (128, 512, 512)
   - 体素间距: (sz, sy, sx) = (1.5, 1.0, 1.0) mm

2. NIfTI文件
   - nibabel加载: (X, Y, Z) - RAS坐标系
   - 转置后: (Z, Y, X) - 标准医学影像坐标系

3. DICOM文件
   - 单张切片: pixel_array.shape = (rows, cols) = (Y, X)
   - 堆叠后: (Z, Y, X) - 无需转置

4. 探测器 (我们的约定)
   - detector_shape = (width, height) = (640, 560)
   - width: 水平方向 (对应X)
   - height: 垂直方向 (对应Z)

5. PyRoNN内部约定
   - detector_shape = (height, width)
   - 使用 swap_detector_axis=True 进行转换

6. 3D Sinogram (我们的约定)
   - 形状: (n_projections, detector_width, detector_height)
   - 默认: (360, 640, 560)
   - 索引: sinogram[角度, 探测器列, 探测器行]

7. 2D Sinogram (单切片)
   - 从3D提取: sinogram_2d = sinogram_3d[:, :, z_idx]
   - 形状: (n_projections, detector_width) = (360, 640)

8. 维度对应关系
   体积 (Z, Y, X)  <->  Sinogram (n_proj, width, height)
        Z          <->  height (探测器垂直方向)
        X          <->  width (探测器水平方向)
        Y          <->  投影角度采样

=============================================================================
"""
    print(info)


if __name__ == "__main__":
    print_dimension_info()

    # 测试配置
    vol_cfg = VolumeConfig()
    det_cfg = DetectorConfig()

    print("\n体积配置:")
    print(f"  目标形状: {vol_cfg.target_shape}")
    print(f"  目标间距: {vol_cfg.target_spacing} mm")
    print(f"  HU范围: [{vol_cfg.hu_min}, {vol_cfg.hu_max}]")

    print("\n探测器配置:")
    print(f"  尺寸: {det_cfg.detector_width} x {det_cfg.detector_height}")
    print(f"  投影数: {det_cfg.n_projections}")
    print(f"  SDD: {det_cfg.source_detector_distance} mm")
    print(f"  SID: {det_cfg.source_isocenter_distance} mm")

    print("\n预期Sinogram形状:")
    print(f"  ({det_cfg.n_projections}, {det_cfg.detector_width}, {det_cfg.detector_height})")
