from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from CombinedAttributesAdder import CombinedAttributesAdder

num_pipeline = pipeline.Pipeline([
  ('inputer', SimpleImputer(strategy='median')),
  ('attribs_adder', CombinedAttributesAdder()),
  ('scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

