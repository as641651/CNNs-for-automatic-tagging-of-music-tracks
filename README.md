# CNNs-for-automatic-tagging-of-music-tracks

This project was carried out as a part of my Master thesis - ["Analyses of CNNs for automatic tagging of music tracks"](https://drive.google.com/file/d/1ljEz-KJPpg3KZHLHEh891Tl3DNh2V1Lp/view?usp=sharing). This repository just contains the framework to train CNNs according to experiments mentioned in my thesis. I have a another [repository](https://github.com/as641651/RythmCap) for testing. 

## Using the training framework

Starting point :

```bash
torch-modules/trainval.lua
```

### Processing the dataset 

Requires dataset settings to be described in a json file. Have a look at the [preprocess_settings.json](https://github.com/as641651/CNNs-for-automatic-tagging-of-music-tracks/blob/master/preprocess_settings.json) to find out the accepted key values. The training data should be in the [magna tag a tune](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) dataset format. To process the dataset,
```bash
python preprocess.py --cfg preprocess_settings.json
```

### Training 

Requires a cfg file like [this one](https://github.com/as641651/CNNs-for-automatic-tagging-of-music-tracks/blob/master/torch-modules/models/C5M1S/magna/01.json) where all the parameters are specified. To train,
```bash
cd torch-modules/
th trainval.lua -c <path_to_cfg_file>
```


