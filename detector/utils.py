import os
import pydicom
import nibabel as nib
from PIL import Image
from django.conf import settings

def convert_dicom_to_png(dicom_path, output_name="mri.png"):
    """Convert DICOM to grayscale PNG"""
    try:
        ds = pydicom.dcmread(dicom_path)
        img = Image.fromarray(ds.pixel_array).convert('L')
        output_path = os.path.join(settings.MEDIA_ROOT, output_name)
        img.save(output_path)
        return output_path
    except Exception as e:
        raise ValueError(f"DICOM conversion failed: {str(e)}")

def convert_nifti_to_png(nifti_path, slice_num=50, output_name="mri.png"):
    """Convert NIfTI slice to grayscale PNG"""
    try:
        img_data = nib.load(nifti_path).get_fdata()
        img = Image.fromarray(img_data[:,:,slice_num]).convert('L')
        output_path = os.path.join(settings.MEDIA_ROOT, output_name)
        img.save(output_path)
        return output_path
    except Exception as e:
        raise ValueError(f"NIfTI conversion failed: {str(e)}")