FastCausation
=============

1. Modification on [baseline code](http://www.causality.inf.ethz.ch/CEdata/ce-jarfo-submission.zip)

2. Speed improvement: reduced 72% of running time

3. Test on Windows 7 and [Anaconda](http://repo.continuum.io/archive/Anaconda-1.6.2-Windows-x86_64.exe)

Usage 
-------------

```python
python predict.py ./data ./results
```

Compile cython code if needed
-------------



```bash
setup.py build_ext --inplace
```
