# Methods for obtaining quality training set.
Implementation of three methods stated in http://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Sanchez-PRL.pdf. These methods are designed to obtain quality training set. Thanks to this, especially in case of a dataset with outliers or with mislabbeled samples, classification model is able to obtain better final accuracy.

### Depuration
Depuration function examines training set and for every item in it checks if the item has at least k_prim nearest neighbors with the label the same as the original label of this item. If there is no enough such neighbors, discard this the item from training set.

### kncn_edit (k nearest centroids editing)
kncn_edit function examines training set and for every item in it checks if the prediction for this item, using k nearest centroids method (centroid space created with the rest items in the training set), is the same as the original label of this item.

### iterative kncn_eidt (iterative k nearest centroids editing)
Iterative kncn_edit function works in the same way as the kncn_edit function but in the while loop. It processes the training set until there is no item missclasified (the training set becomes smaller each iteration).

#### Examples
Prectical usage of the stated methods can be seen in the exmaples.py file. Running this file results in comparison of results obtained without and with implemented methods for obtaining quality training set.