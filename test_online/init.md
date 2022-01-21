# Test on the COCO-test server

Run `python test_online.py` to evaluate the rstnet online:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to visual features file (h5py) |
| `--annotation_folder` | Path to m2_annotations |
| `--model_path` | Path to model files |
| `--output_path` | Path to captions files |
