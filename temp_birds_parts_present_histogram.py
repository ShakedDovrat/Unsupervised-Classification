import numpy as np
import pandas as pd
import tqdm

part_locs_path = r"E:\datasets\Caltech-UCSD-Birds-200-2011\CUB_200_2011\parts\part_locs.txt"
part_locs_pd = pd.read_csv(part_locs_path, delim_whitespace=True, header=None)
part_locs_pd.columns = ['image_id', 'part_id', 'x', 'y', 'visible']

per_image = part_locs_pd.groupby('image_id')

part_names = part_locs_pd['part_id'].unique()
num_parts = len(part_names)
image_ids = []
are_parts_visible = []

for image_id, row_indices in tqdm.tqdm(per_image.groups.items(), desc='Extracting attributes'):
    curr_pd = part_locs_pd.iloc[row_indices, :]
    visible_part_names = curr_pd['part_id'].to_numpy()[curr_pd['visible'].to_numpy().astype(np.bool)]
    visible_part_indices = np.isin(part_names, visible_part_names)
    image_ids.append(image_id)
    are_parts_visible.append(visible_part_indices)
# for part_id in part_locs_pd['part_id'].unique():

image_ids = np.array(image_ids)
are_parts_visible = np.stack(are_parts_visible)

eyes_part_names = (7, 11)
legs_part_names = (8, 12)
wings_part_names = (9, 13)

organ_names = ['eyes', 'legs', 'wings']
organ_part_names = (eyes_part_names, legs_part_names, wings_part_names)
result = []
result_names = ['both', 'none', 'xor']
for organ_name, curr_part_names in zip(organ_names, organ_part_names):
    left_right_visible = are_parts_visible[:, np.isin(part_names, curr_part_names)]
    both = left_right_visible.all(axis=1)
    none = ~left_right_visible.any(axis=1)
    xor = ~none & ~both
    assert (both | none | xor).all()
    assert ~((both & none & xor).any())
    result.append((both.mean(), none.mean(), xor.mean()))

result = np.stack(result)
result_df = pd.DataFrame(data=result, index=organ_names, columns=result_names)
print(result_df*100)
pass
