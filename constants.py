mode = 'docker container'

if mode == 'docker container':
    RAW_TRAIN_PATH = '/home/code/data/train_go05W65.csv'
    RAW_TEST_PATH = '/home/code/data/test_VkM91FT.csv'
    TRAIN_REPORT_PATH = '/home/code/reports/train_report.html'
    TEST_REPORT_PATH = '/home/code/reports/test_report.html'
    PROCESSED_TRAIN_PATH = '/home/code/data/processed_train.csv'
    PROCESSED_TEST_PATH = '/home/code/data/processed_test.csv'
    ARTIFACTS_PATH = '/home/code/artifacts/'
    PREDICTION_FILE_PATH = '/home/code/data/prediction_file.csv'

elif mode == 'local':
    RAW_TRAIN_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/data/train_go05W65.csv'
    RAW_TEST_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/data/test_VkM91FT.csv'
    TRAIN_REPORT_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/reports/train_report.html'
    TEST_REPORT_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/reports/test_report.html'
    PROCESSED_TRAIN_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/data/processed_train.csv'
    PROCESSED_TEST_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/data/processed_test.csv'
    ARTIFACTS_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/artifacts/'
    PREDICTION_FILE_PATH = '/Users/vishwanathprudhivi/Desktop/Code/amex_analytics_vidhya/analytics_vidhya_amex_hackathon/data/prediction_file.csv'

CATEGORICAL_FEATURES = ['gender','city_category','customer_category']
NUMERICAL_FEATURES = ['age','vintage','is_active']
TARGET = 'target'
