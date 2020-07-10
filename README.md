# covid19_npis_europe


## First steps

Install all required packages
```
pip intall -r requirements.txt
```


## Frequent Problems

### Is my gpu working?
If we want to know if our GPU is working, we can run `tf.config.list_physical_devices()`.

If the returned array is empty or only shows cpu nodes One can try to follow this guide:
https://www.tensorflow.org/install/gpu

Is quite tedious tho and requires a nvidia developer account.

This is optional but (may?) improve performance.