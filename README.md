# ASR (DeepSpeech) project

## Installation guide

Clone repository
   ```shell
   git clone https://github.com/KemmerEdition/asr_project_template.git
   ```
Maybe then you need to change directory

   ```shell
   cd /content/asr_project_template
   ```
Download requirements and checkpoint of my model
   ```shell
   pip install -r requirements.txt
   ```
   ```shell
   !conda install -y gdown
   !gdown --id 1IaNeaaOSjUYW8cmKNpRkkkCwxaz5Bcid
   ```
## Test
   ```shell
   python -m test \
      -c default_test_config.json \
      -r checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
