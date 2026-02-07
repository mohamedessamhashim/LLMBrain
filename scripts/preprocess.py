#!/usr/bin/env python
"""
MRI Preprocessing Pipeline for UCSF-PDGM

Steps:
1. N4 Bias Field Correction (SimpleITK)
2. Registration to MNI template (1mm isotropic)
3. Brain Extraction (HD-BET)
4. Intensity Normalization (z-score)

Usage:
    python scripts/preprocess.py \
        --input_dir data/raw/UCSF-PDGM \
        --output_dir data/processed \
        --template data/templates/mni_t1.nii.gz \
        --device cuda:0
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess UCSF-PDGM MRI data")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing raw UCSF-PDGM data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Path to MNI template for registration (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device for HD-BET (default: 0)",
    )
    parser.add_argument(
        "--skip_brain_extraction",
        action="store_true",
        help="Skip brain extraction step",
    )
    parser.add_argument(
        "--skip_registration",
        action="store_true",
        help="Skip registration step (still resamples to 1mm isotropic)",
    )
    return parser.parse_args()


def n4_bias_correction(image: sitk.Image) -> sitk.Image:
    """Apply N4 bias field correction.

    Args:
        image: Input SimpleITK image.

    Returns:
        Bias-corrected image.
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    corrected = sitk.N4BiasFieldCorrection(image)
    return corrected


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Resample image to isotropic voxel spacing.

    Args:
        image: Input SimpleITK image.
        target_spacing: Target voxel spacing in mm.
        interpolator: SimpleITK interpolator type.

    Returns:
        Resampled image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0)
    resample.SetOutputPixelType(sitk.sitkFloat32)

    return resample.Execute(image)


def register_to_template(
    moving: sitk.Image,
    fixed: sitk.Image,
) -> Tuple[sitk.Image, sitk.Transform]:
    """Register moving image to fixed template using rigid registration.

    Args:
        moving: Moving image to register.
        fixed: Fixed template image.

    Returns:
        Tuple of (registered image, transform).
    """
    # Initialize transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Multi-resolution registration
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform)

    final_transform = registration.Execute(fixed, moving)

    # Resample moving image
    registered = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    return registered, final_transform


def apply_transform(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """Apply a transform to an image.

    Args:
        image: Image to transform.
        reference: Reference image for output geometry.
        transform: Transform to apply.
        interpolator: SimpleITK interpolator type.

    Returns:
        Transformed image.
    """
    return sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        0.0,
        image.GetPixelID(),
    )


def z_score_normalize(image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalize image intensities.

    Args:
        image: Input image array.
        mask: Optional brain mask for computing statistics.

    Returns:
        Normalized image array.
    """
    if mask is not None:
        brain_voxels = image[mask > 0]
    else:
        brain_voxels = image[image > 0]

    if len(brain_voxels) == 0:
        return image

    mean = brain_voxels.mean()
    std = brain_voxels.std()

    if std > 0:
        normalized = (image - mean) / std
    else:
        normalized = image - mean

    return normalized


def find_modality_files(subject_dir: Path) -> dict:
    """Find MRI modality files in a subject directory.

    Args:
        subject_dir: Path to subject directory.

    Returns:
        Dictionary mapping modality names to file paths.
    """
    modalities = {}
    patterns = {
        "t1": ["*_t1.nii.gz", "*_T1.nii.gz", "*t1*.nii.gz"],
        "t1ce": ["*_t1ce.nii.gz", "*_t1gd.nii.gz", "*_T1ce.nii.gz", "*_T1GD.nii.gz"],
        "t2": ["*_t2.nii.gz", "*_T2.nii.gz", "*t2*.nii.gz"],
        "flair": ["*_flair.nii.gz", "*_FLAIR.nii.gz", "*flair*.nii.gz"],
        "seg": ["*_seg.nii.gz", "*_SEG.nii.gz", "*seg*.nii.gz", "*mask*.nii.gz"],
    }

    for modality, pats in patterns.items():
        for pat in pats:
            matches = list(subject_dir.glob(pat))
            # Filter out matches that are actually other modalities
            if modality == "t1":
                matches = [m for m in matches if "t1ce" not in m.name.lower() and "t1gd" not in m.name.lower()]
            if matches:
                modalities[modality] = matches[0]
                break

    return modalities


def process_subject(
    subject_dir: Path,
    output_dir: Path,
    template: Optional[sitk.Image] = None,
    device: str = "0",
    skip_brain_extraction: bool = False,
    skip_registration: bool = False,
) -> bool:
    """Process a single subject.

    Args:
        subject_dir: Path to subject directory.
        output_dir: Output directory.
        template: Optional template image for registration.
        device: GPU device for HD-BET.
        skip_brain_extraction: Skip brain extraction step.
        skip_registration: Skip registration step.

    Returns:
        True if processing succeeded, False otherwise.
    """
    subject_id = subject_dir.name
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find modality files
    modalities = find_modality_files(subject_dir)
    required = ["t1", "t1ce", "t2", "flair"]

    # Check if all required modalities exist
    missing = [m for m in required if m not in modalities]
    if missing:
        print(f"  Skipping {subject_id}: missing modalities {missing}")
        return False

    # Check if segmentation exists
    if "seg" not in modalities:
        print(f"  Skipping {subject_id}: missing segmentation")
        return False

    try:
        # Load FLAIR for registration reference (typically best contrast)
        flair_img = sitk.ReadImage(str(modalities["flair"]), sitk.sitkFloat32)

        # Apply N4 bias correction to FLAIR
        flair_corrected = n4_bias_correction(flair_img)

        # Determine reference image and transform
        if template is not None and not skip_registration:
            # Register to template
            template_resampled = resample_to_isotropic(template)
            flair_registered, transform = register_to_template(flair_corrected, template_resampled)
            reference = template_resampled
        else:
            # Just resample to isotropic
            flair_registered = resample_to_isotropic(flair_corrected)
            transform = sitk.Transform()
            reference = flair_registered

        # Process each modality
        modality_order = ["t1", "t1ce", "t2", "flair"]
        for i, mod in enumerate(modality_order):
            if mod == "flair":
                # Already processed
                processed = flair_registered
            else:
                # Load and correct
                img = sitk.ReadImage(str(modalities[mod]), sitk.sitkFloat32)
                img_corrected = n4_bias_correction(img)

                if template is not None and not skip_registration:
                    # Apply same transform
                    processed = apply_transform(img_corrected, reference, transform)
                else:
                    processed = resample_to_isotropic(img_corrected)

            # Save
            output_path = images_dir / f"{subject_id}_000{i}.nii.gz"
            sitk.WriteImage(processed, str(output_path))

        # Process segmentation (nearest neighbor interpolation)
        seg_img = sitk.ReadImage(str(modalities["seg"]), sitk.sitkFloat32)
        if template is not None and not skip_registration:
            seg_processed = apply_transform(
                seg_img, reference, transform, sitk.sitkNearestNeighbor
            )
        else:
            seg_processed = resample_to_isotropic(seg_img, interpolator=sitk.sitkNearestNeighbor)

        seg_output_path = labels_dir / f"{subject_id}.nii.gz"
        sitk.WriteImage(seg_processed, str(seg_output_path))

        return True

    except Exception as e:
        print(f"  Error processing {subject_id}: {e}")
        return False


def run_brain_extraction(output_dir: Path, device: str = "0") -> None:
    """Run HD-BET brain extraction on all processed images.

    Args:
        output_dir: Directory containing preprocessed images.
        device: GPU device for HD-BET.
    """
    try:
        from HD_BET.hd_bet import hd_bet
    except ImportError:
        print("HD-BET not installed. Skipping brain extraction.")
        print("Install with: pip install git+https://github.com/MIC-DKFZ/HD-BET.git")
        return

    images_dir = output_dir / "images"
    temp_dir = output_dir / "temp_hdbet"
    temp_dir.mkdir(exist_ok=True)

    # HD-BET processes directories, so we need to group by subject
    subjects = set()
    for f in images_dir.glob("*_0000.nii.gz"):
        subjects.add(f.stem.replace("_0000", ""))

    print(f"Running brain extraction on {len(subjects)} subjects...")

    for subject_id in tqdm(subjects):
        # Use FLAIR (0003) for brain extraction mask
        flair_path = images_dir / f"{subject_id}_0003.nii.gz"
        if not flair_path.exists():
            continue

        try:
            # Run HD-BET on FLAIR
            hd_bet(
                str(flair_path),
                str(temp_dir / f"{subject_id}_mask"),
                device=device,
                mode="fast",
                tta=0,
            )

            # Load mask
            mask_path = temp_dir / f"{subject_id}_mask_mask.nii.gz"
            if not mask_path.exists():
                continue

            mask = nib.load(mask_path).get_fdata()

            # Apply mask to all modalities
            for i in range(4):
                img_path = images_dir / f"{subject_id}_000{i}.nii.gz"
                nii = nib.load(img_path)
                data = nii.get_fdata()
                data_masked = data * mask
                nib.save(
                    nib.Nifti1Image(data_masked, nii.affine, nii.header),
                    img_path,
                )

        except Exception as e:
            print(f"  Brain extraction failed for {subject_id}: {e}")

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template if provided
    template = None
    if args.template is not None:
        template_path = Path(args.template)
        if template_path.exists():
            template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
            print(f"Using template: {template_path}")
        else:
            print(f"Template not found: {template_path}")

    # Find all subject directories
    subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subject_dirs)} subject directories")

    # Process each subject
    successful = 0
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        if process_subject(
            subject_dir,
            output_dir,
            template,
            args.device,
            args.skip_brain_extraction,
            args.skip_registration,
        ):
            successful += 1

    print(f"Successfully processed {successful}/{len(subject_dirs)} subjects")

    # Run brain extraction if not skipped
    if not args.skip_brain_extraction:
        run_brain_extraction(output_dir, args.device)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
