# Codebase for Image-Based Query Search Engine via Deep Learning

We have designed and implemented an image-based query search engine that strikes a good balance between efficiency and accuracy. Users can submit any image they want to search for, and the engine will return similar images in a custom database. The main components of the system are shown in the following diagram:
![pipeline](https://user-images.githubusercontent.com/76591676/181504716-76a20f35-3485-4489-8f81-1104651e2c05.png)

This work is a combination of three master's thesis projects. Welcome to check out our theses via the following links:
- [ ] Yanan Hu (Feature extraction): 
- [x] Yuanyuan Yao (Nearest neighbour search): https://repository.tudelft.nl/islandora/object/uuid:4a2c9c6f-b2b8-41d6-9b70-69c4f246c964
- [x] Qi Zhang (Re-ranking): https://repository.tudelft.nl/islandora/object/uuid%3A32e02913-ba0d-446a-9807-1129ba4a314b
- [ ] *Add the link to the publication if the work is eventually published*
- [ ] *Add the link to the interface if the work is integrated into [EHM](https://engineeringhistoricalmemory.com/Aggregators.php)*


## To simply use the pretrained models directly:
- [ ] Add a directory tree
1. [Create and activate a virtual python environment](https://docs.python.org/3/library/venv.html)
2. Install packages using requirements.txt  
   `pip install -r requirements.txt`  
   Faiss needs to be installed manually: `pip install faiss-gpu`  
   The torch version and cuda version should be compatible
3. Download the pretrained network from https://drive.google.com/drive/folders/1JbGNvQgqKm7GiUvOqw1DSncSVR3k0xbm?usp=sharing and save it under data/networks
4. Change the paths in function `extr_selfmade_dataset` (src/networks/imageretrievalnet.py) to the paths of your datasets (which are just folers contain jpg images)
5. [Create symbolic link](https://www.freecodecamp.org/news/symlink-tutorial-in-linux-how-to-create-and-remove-a-symbolic-link/) to map your datasets under static/test/
6. Run offline.py to extract and save the features of images  
   ```bash
   python3 -m src.offline --datasets 'YOUR_DATASET_1, YOUR_DATASET_2, …, YOUR_DATASET_N' --gpu '0' --network 'resnet101-solar-best.pth' --K-nearest-neighbour 100
   ```
   - The datasets will be merged to be your database. Given a query image, the engine will find the most similar images in the database.
   - If the database is large-scale (>100k), then you may need to use approximate nearest neighbour search methods, e.g., ANNOY.  Select it by adding `--matching_method 'ANNOY' --ifgenerate` after the original command. It is normal that offline.py runs for a long time (even for days if the database is million-scale and HNSW or PQ_HNSW is chosen).
   - Still, pay attention to the paths of the outputs. You can find and modify the settings in functions save_path_feature and load_path_feature (src/utils/general.py).
7. Run online.py  
   ```bash
   python3 -m src.online --datasets 'YOUR_DATASET_1, YOUR_DATASET_2, …, YOUR_DATASET_N' --gpu '0' --network 'resnet101-solar-best.pth' --K-nearest-neighbour 100
   ```
   - The datasets and network should be exactly the same as the ones you choose when running offline.py
   - Use neighbour search methods if necessary. But do not include `--ifgenerate` since the required data/structures have been generated.
   - After running a link will appear, click and operate on the GUI interface. Upload the query image and wait for the results.

## If you want to tweak the model or reproduce our results:


<details><summary><b>Training</b></summary>

<p>

We have already trained the model "Resnet101-solar-best" with good results, which is stored at https://drive.google.com/drive/folders/1JbGNvQgqKm7GiUvOqw1DSncSVR3k0xbm?usp=sharing. We recommend that you use ths pre-trained model. If you want to use our pre-trined model, download it and place it in ~/data/networks/ , then skip the following instructions directly to next part.

If you wish to retrain the model yourself, the Example training script is located in ~/src/main_train.py

To train the model, you should firstly make sure you have downloaded the training datasets Sfm120k or GoogleLandmarksv2 in  ~/data/train/, then you can start the training with the settings described in the paper by running

```ruby
   python3 -m main_train [-h] [--training-dataset DATASET] [--no-val]
                [--test-datasets DATASETS] [--test-whiten DATASET]
                [--test-freq N] [--arch ARCH] [--pool POOL]
                [--local-whitening] [--regional] [--whitening]
                [--not-pretrained] [--loss LOSS] [--loss-margin LM]
                [--image-size N] [--neg-num N] [--query-size N]
                [--pool-size N] [--gpu-id N] [--workers N] [--epochs N]
                [--batch-size N] [--optimizer OPTIMIZER] [--lr LR] [--ld LD]
                [--soa] [--weight-decay W] [--soa-layers N] [--sos] [--lambda N] 
                [--print-freq N] [--flatten-desc]
                EXPORT_DIR
```
</p>
</details>

<details><summary><b>Test</b></summary>
<p>
Firstly, please make sure you have downloaded the test datasets and put them under ~/data/test/.
Then you can start retrieval tests as following:


### Testing on R-Oxford, R-Paris

```ruby
   python3 -m ~src.main_retrieve
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.
### Testing with the extra 1-million distractors
```ruby
   python3 -m ~src.extract_1m
   python3 -m ~src.test_1m
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.
### Testing on Custom
```ruby
   python3 -m ~src.test_custom
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.

### Testing on GoogleLandmarks v2 test
```ruby
   python3 -m ~src.test_GLM
```
You can view the automatically generated example ranking images in ~outputs/ranks/. Also, the extracted feature files are automatically saved in ~outputs/features/.

### Testing re-ranking methods
You can use three re-ranking methods (QGE, SAHA, and LoFTR) in any datasets in the following python files:
```ruby
   python3 -m ~src.test_extracted # This is an example of our pipeline. You can test any datasets with this file.
   python3 -m ~src.server   # This is our pipeline with GUI.
```
These two python files can help you to use re-ranking.  

By these files, you can test extracted features from any dataset. You can put preextracted features under this path: src/outputs/. And please unzip the file (utils_files.zip) in "src/utils/" before using.  

And please check paths in "test_extracted.py", "server.py", and "Reranking" (under "src/utils/") before using. You need to set your own paths on a Linux server or your local computer.   

The pretrained feature extraction weight: https://drive.google.com/file/d/1fylhFYW0vYIBpYts_bx4IMiIPL34V5Yb/view?usp=sharing   

You can put rhe weight under this path: src/EXPORT_DIR_QZ/resnet101-gem-w-tri/

To test re-ranking methods, you can use the following api in the aforementioned two files:

For QGE:
```ruby
QGE(ranks, qvecs, vecs, dataset, gnd, query_num, cache_dir, gnd_path2, RW, AQE)  
```
For SAHA: 
```ruby
sift_online(query_num, qimages, sift_q_main_path, images, sift_g_main_path, ranks, dataset, gnd)  
```
For LoFTR: 
```ruby
loftr(loftr_weight_path, query_num, qimages, ranks, images, dataset, gnd)  
```
If you want to use LoFTR, you need to download the pretrained LoFTR weight from: https://github.com/zju3dv/LoFTR  
You can put the LoFTR weight under this path: src/utils/weights/  

You can find detailed annotations about how to use these re-ranking methods in Reranking.py, test_extracted.py and server.py.  

</p>
</details>
<details><summary><b>Retrieval Engine</b></summary>
<p>


</p>
</details>
