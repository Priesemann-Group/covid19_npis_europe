# covid19_npis_europe

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## First steps
Initialize submodule while cloning
```
git clone --recurse-submodules https://github.com/Priesemann-Group/covid19_npis_europe.git
```

Install all required packages
```
pip install -r requirements.txt
```


## Frequent Problems

### Is my gpu working?
If we want to know if our GPU is working, we can run `tf.config.list_physical_devices()`.

If the returned array is empty or only shows cpu nodes, one can try to follow this guide:
https://www.tensorflow.org/install/gpu

Is quite tedious tho and requires a nvidia developer account.

This is optional but (may?) improve performance.
