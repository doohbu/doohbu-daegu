# Colab VM 에서: 공식 example.py 와 같은 단계로 입력 한 장 → GLB. 기록은 /content/run.log
import os, sys, time
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel
t0 = time.time()
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
print('불러오기', round(time.time() - t0), '초', flush=True)
image = Image.open(sys.argv[1])
t1 = time.time()
torch.manual_seed(1)
mesh = pipeline.run(image)[0]
print('생성', round(time.time() - t1), '초, 최대 GPU', round(torch.cuda.max_memory_allocated() / 2**30, 1), 'GiB', flush=True)
mesh.simplify(16777216)
glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                                 attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                 decimation_target=1000000, texture_size=4096, remesh=True, remesh_band=1, remesh_project=0, verbose=True)
glb.export(sys.argv[2], extension_webp=True)
print('끝', round(time.time() - t0), '초', flush=True)
