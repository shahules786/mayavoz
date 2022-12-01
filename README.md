<p align="center">
  <img src="https://user-images.githubusercontent.com/25312635/195514652-e4526cd1-1177-48e9-a80d-c8bfdb95d35f.png" />
</p>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/shahules786/enhancer/Enhancer)
![GitHub](https://img.shields.io/github/license/shahules786/enhancer)
![GitHub issues](https://img.shields.io/github/issues/shahules786/enhancer?logo=GitHub)
![GitHub Repo stars](https://img.shields.io/github/stars/shahules786/enhancer?style=social)
![GitHub all releases](https://img.shields.io/github/downloads/shahules786/enhancer/total)

mayavoz is a Pytorch-based opensource toolkit for speech enhancement. It is designed to save time for audio practioners & researchers. Is provides easy to use pretrained audio enhancement models and facilitates highly customisable model training.

| **[Quick Start](#quick-start-fire)** | **[Installation](#installation)** | **[Tutorials](https://github.com/shahules786/enhancer/tree/main/notebooks)** | **[Available Recipes](#recipes)** | **[Demo](#demo)**
## Key features :key:

* Various pretrained models nicely integrated with huggingface 	:hugs: that users can select and use without any hastle.
* :package: Ability to train and validation your own custom speech enhancement models with just under 10 lines of code!
* :magic_wand: A command line tool that facilitates training of highly customisable speech enhacement models from the terminal itself!
* :zap: Supports multi-gpu training integrated with Pytorch Lightning.


## Demo

Noisy audio followed by enhanced audio.

https://user-images.githubusercontent.com/25312635/203756185-737557f4-6e21-4146-aa2c-95da69d0de4c.mp4



## Quick Start :fire:
``` python
from mayavoz.models import Mayamodel

model = Mayamodel.from_pretrained("shahules786/mayavoz-waveunet-valentini-28spk")
model.enhance("noisy_audio.wav")
```

## Recipes

| Model     | Dataset      | STOI    | PESQ  | URL                           |
| :---:     |  :---:       | :---:   | :---: | :---:                         |
| WaveUnet  | Valentini-28spk   | 0.836   | 2.78  |  shahules786/mayavoz-waveunet-valentini-28spk      |
| Demucs    | Valentini-28spk   | 0.961   | 2.56  |  shahules786/mayavoz-demucs-valentini-28spk       |
| DCCRN     | Valentini-28spk   | 0.724   | 2.55  |  shahules786/mayavoz-dccrn-valentini-28spk         |
| Demucs     | MS-SNSD-20hrs  | 0.56 | 1.26  | shahules786/mayavoz-demucs-ms-snsd-20       |

Test scores are based on respective test set associated with train dataset.

**See [tutorials](/notebooks/) to train your custom model**

## Installation
Only Python 3.8+ is officially supported (though it might work with Python 3.7)

- With Pypi
```
pip install mayavoz
```

- With conda

```
conda env create -f environment.yml
conda activate mayavoz
```

- From source code
```
git clone url
cd mayavoz
pip install -e .
```

## Support

For commercial enquiries and scientific consulting, please [contact me](https://shahules786.github.io/).

### Acknowledgements
Sincere gratitude to [AMPLYFI](https://amplyfi.com/) for supporting this project.
