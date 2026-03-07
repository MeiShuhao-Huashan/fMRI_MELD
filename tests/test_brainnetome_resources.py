import numpy as np

from meld_fmri.fmri_gcn.atlas import load_atlas_brainnetome


def test_brainnetome_resources_load() -> None:
    atlas = load_atlas_brainnetome()
    assert atlas.name == "brainnetome"
    assert len(atlas.parcel_names) == 105

    assert atlas.fsaverage_sym_lh_parcel_index.ndim == 1
    assert atlas.fsaverage_sym_lh_parcel_index.dtype == np.int32

    assert atlas.fslr32k_lh_parcel_index.shape == (32492,)
    assert atlas.fslr32k_rh_parcel_index.shape == (32492,)

    assert atlas.parcel_centroids_fsavg_sym_lh.shape == (105, 3)
    assert atlas.cortex_mask_fsavg_sym_lh.ndim == 1

