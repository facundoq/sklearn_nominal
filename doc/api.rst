.. _api:

#############
API Reference
#############

This is an example on how to document the API of your own project.

.. currentmodule:: sklearn_nominal

`scikit-learn` compatible classifiers
======================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TreeClassifier
   NaiveBayesClassifier
   ZeroRClassifier
   OneRClassifier
   CN2Classifier
   PRISMClassifier

`scikit-learn` compatible regressors
=====================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   TreeRegressor
   ZeroRRegressor
   OneRRegressor
   CN2Regressor

Base classes for some models
============================

`BaseTree` for example, defines the pruning parameters for both `TreeRegressor`
and `TreeClassifier`. Same thing with

.. autosummary::
   :toctree: generated/
   :template: class.rst

   sklearn.tree_base.BaseTree
   sklearn.nominal_model.NominalModel
   sklearn.nominal_model.NominalClassifier
   sklearn.nominal_model.NominalRegressor
