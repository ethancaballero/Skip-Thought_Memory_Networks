# stmn

Question Answering system from stmn.

## Requirements:
- Python 2.7
- Theano 0.7 (with floatX set to float64)
- recent [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
- [Keras](https://github.com/fchollet/keras) 
- [NLTK 3](http://www.nltk.org/)
- [scikit-learn](http://scikit-learn.org/stable/index.html) (optional)
- [gensim](https://radimrehurek.com/gensim/) (optional)



#
download data to corresponding directories in stmn folder:
```
cd (top_directory_of_folder)
mkdir data
cd data

curl -O http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
curl -O http://www.cs.toronto.edu/~rkiros/models/utable.npy
curl -O http://www.cs.toronto.edu/~rkiros/models/btable.npy
curl -O http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
curl -O http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
curl -O http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
curl -O http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

##To Run:
```
python wmemnn.py babi/qa5_three-arg-relations_train.txt
```
if minimal computing resources are available:
```
python wmemnn.py babi/qa5_short_train.txt
```
some others available in:
```
python wmemnn.py babi/en/~~~.txt
```

##Note
to redownload bAbi manually
```
cd (top_directory_of_folder)
curl -O http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar -xvzf http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
mv tasks_1-20_v1-2 babi
```

## Acknowledgements
based on Pararth Shah's [qa-memnn](https://github.com/pararthshah/qa-memnn) implementation and Ryan Kiros' [skip-thoughts](https://github.com/ryankiros/skip-thoughts) implementation.


## License

MIT
