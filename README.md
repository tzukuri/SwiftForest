# Swift Forest

An implementation of: Extremely randomized trees (Geurts et al. 2006)
http://www.montefiore.ulg.ac.be/%7Eernst/uploads/news/id63/extremely-randomized-trees.pdf

Decision trees are a simple but powerful machine learning technique used in supervised
classification. While the trees themselves are easy to comprehend after training, some
implementations are prone to overfitting and the training process can be slow. Extremely
randomised trees are an easy implementation, but are state of the art, efficient to train,
and strongly resistent to overfitting.

Swift Forest deals with training examples of real valued features, and discrete output
classes (i.e supervised classification). Individual trees or a forest (ensemble) can
be constructed, and there are only 3 parameters:

* number of trees (if creating a forest, defaults to 100)
* minimum number of examples a node can have if it's to split (defaults to 2)
* number of features to randomly select during a split (defaults to sqrt(features))

## Licence

![Public Domain](http://i.creativecommons.org/p/zero/1.0/88x31.png)

Public Domain: To the extent possible under law, Tzukuri Pty Ltd has waived all copyright and related or neighbouring rights to SwiftForest. This work is published from: Australia.

https://creativecommons.org/publicdomain/zero/1.0/

