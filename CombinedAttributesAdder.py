from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

from fetch_housing_data import *

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
       def __init__(self, add_bedrooms_per_room = True): # sem *args ou **kargs
           self.add_bedrooms_per_room = add_bedrooms_per_room
       def fit(self, X, y=None):
           return self  # nothing else to do
       def transform(self, X, y=None):
           rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
           population_per_household = X[:, population_ix] / X[:, household_ix]
           if self.add_bedrooms_per_room:
               bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
               return np.c_[X, rooms_per_household, population_per_household,
                            bedrooms_per_room]
               return np.c_[X, rooms_per_household, population_per_household]

housing = load_housing_data()
housing.head()

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)