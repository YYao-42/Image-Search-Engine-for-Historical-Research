## Documents

To better inform the members of the project progress and understand each other's work, we will keep updating two online documents as the project progresses. 

- Project documentation: https://hackmd.io/@yyao/H17LYJEKF/edit

- Appendix: https://hackmd.io/@yyao/rk4UNPIFY/edit

**Please always update them when you make modifications to the codes.**

## Collaborative Workflow

#### Basic Shared Repository Workflow

- update your local repo with `git pull origin master`,
- create a working branch with `git checkout -b MyNewBranch`
- make your changes on your branch and stage them with `git add`,
- commit your changes locally with `git commit -m "description of your commit"`, and
- upload the changes (including your new branch) to GitHub with `git push origin MyNewBranch`
- go to the main repo on GitHub where you should now see your new branch
- click on your branch name
- click on “Pull Request” button (URC)
- click on “Send Pull Request”

Reference: https://uoftcoders.github.io/studyGroup/lessons/git/collaboration/lesson/

#### Fork and Pull Workflow (Recommended)

https://gist.github.com/Chaser324/ce0505fbed06b947d962

#### The summary of the previous work(Bachelar project and Extra project)

https://docs.google.com/document/d/1mE22UhICK-X7jXdfCJ-9LtQ6EoYRkYPD6eJ_iM178j8/edit?usp=sharing

## The summary of group progress 
https://hackmd.io/R3bNtFL2Q0GVv9HIbn62Vw

## The pipeline
![pipeline](https://user-images.githubusercontent.com/76591676/181504716-76a20f35-3485-4489-8f81-1104651e2c05.png)

## Usage
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
Firstly， please make sure you have downloaded the test datasets and put them under ~/data/test/.
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
   python3 -m ~src.test_extract
   python3 -m ~src.server
```
In these files, you can test extracted features from any dataset. These two python files can help you to use re-ranking. And please unzip the file in "src/utils" before using.
The pretrained feature extraction weight:https://drive.google.com/file/d/1fylhFYW0vYIBpYts_bx4IMiIPL34V5Yb/view?usp=sharing
   
To test re-ranking methods, you can use the following api:
   
For QGE: QGE(ranks, qvecs, vecs, dataset, gnd, query_num, cache_dir, gnd_path2, RW, AQE)  
For SAHA: sift_online(query_num, qimages, sift_q_main_path, images, sift_g_main_path, ranks, dataset, gnd)  
For LoFTR: loftr(loftr_weight_path, query_num, qimages, ranks, images, dataset, gnd).  
If you want to use LoFTR, you need to download the pretrained LoFTR weight from: https://github.com/zju3dv/LoFTR

</p>
</details>
<details><summary><b>Retrieval Engine</b></summary>
<p>


</p>
</details>
