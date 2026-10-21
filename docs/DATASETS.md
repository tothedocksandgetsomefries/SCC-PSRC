# Datasets

The experiments use the following vector datasets. Raw datasets are obtained separately from their providers.

| Dataset | Database vectors | Dimension | Description and source |
|---|---:|---:|---|
| SIFT1M | 1,000,000 | 128 | Image descriptors from the [TexMex benchmark collection](http://corpus-texmex.irisa.fr/). Its official SIFT1M archive is [sift.tar.gz](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz). |
| DEEP1M | 1,000,000 | 96 | Deep image descriptors used as a one-million-vector benchmark. |
| GLOVE100 | 1,183,514 | 100 | GloVe word vectors. ANN-Benchmarks provides the corresponding [100-dimensional benchmark HDF5](https://ann-benchmarks.com/glove-100-angular.hdf5); its [dataset preparation code](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py) uses the Twitter 27B vectors from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/). |
| NYTimes | 289,761 | 256 | Document vectors distributed by ANN-Benchmarks as [nytimes-256-angular.hdf5](https://ann-benchmarks.com/nytimes-256-angular.hdf5). The project excludes 239 zero-valued database vectors from its 290,000-vector database. |
| W3 | 100,000 | 4096 | arXiv abstract embeddings and row-aligned metadata from [SPCL/arxiv-for-fanns-medium](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium). W3 is the name used in this project. |

GLOVE100 and NYTimes vectors are normalized in FP64 and stored in FP32. The project applies no additional normalization to SIFT1M, DEEP1M or W3. The NYTimes input is the ANN-Benchmarks vector representation, rather than the original bag-of-words collection.

For W3, [DATA_SOURCES.json](../manifests/DATA_SOURCES.json) records the fixed source revision `b6ae30c917d155b4fa4721a3f1cedb3819fed14b`, download URLs, file sizes and hashes. Keep database vectors and metadata in their original row order.

Access terms are provided on the source pages.
