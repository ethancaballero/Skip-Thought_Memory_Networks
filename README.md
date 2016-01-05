# stmn

Question Answering system.

## Dependencies:
* Python 2.7
* Theano 0.7 (with floatX set to float64)
* recent [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK 3](http://www.nltk.org/)
* [Keras](https://github.com/fchollet/keras) 
* [gensim](https://radimrehurek.com/gensim/) 

#
```
cd 
mkdir data
cd data

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

```
mkdir en
cd en

http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
```
```
python wmemnn.py en/qa5_three-arg-relations_train.txt
python wmemnn.py en/qa5_short_train.txt
```

## Acknowledgements


## License

MIT
