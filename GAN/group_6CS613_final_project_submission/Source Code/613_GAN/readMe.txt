## Install dependencies

Python3 and Pip is required to install the dependencies.

```bash
pip install -r requirements.txt
```
If you run main.py it will train the 3 separate GANs for the different frames
Note that this will take a while (>6 hours per model). The outputs will be saved every 1000 epochs into the output folder.
If you want to train just one model, comment out the other two lines at the end of the file

In order to generate the enhanced data comparison image, run enhance.py in the ImageEnhancement

In order to generate FID data run evaluate.py in the ImageEvaluation directory
