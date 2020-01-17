from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse_loss

class Trainer:

    def __init__(self, degree=6):
        
        self.regressor = LinearRegression()
        self.feature_poly = PolynomialFeatures(degree=degree)
        
    def train_with_polynomial_regression(self, feed, target):
        # subsets        
        train_feed, test_feed, train_target, test_target = tts(feed, target, test_size=0.2, random_state=0)

        print('Learning: Polynomial Regression')
        
        train_feed_poly = self.feature_poly.fit_transform(train_feed)        
        self.regressor.fit(train_feed_poly, train_target)

        print('\nEnd')
        print('')

        self.test_with_polynomial_regression(test_feed, test_target)
    
    def test_with_polynomial_regression(self, feed, target):
        print('Checking accuracy...')
        print('')

        feed_poly = self.feature_poly.fit_transform(feed)
        test_predictions = self.regressor.predict(feed_poly)
                
        loss = mse_loss(target, test_predictions)        
        
        for i, test in enumerate(test_predictions):
            print('{} - {} => {} {}'.format(i, test, target[i], test - target[i]))  

        print('')

        print('loss')
        print(loss)
        print('')        
        
        print('End')
        
    