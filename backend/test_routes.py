import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import unittest
from backend.main import create_app

class TestRoutes(unittest.TestCase):
    def setUp(self):
        # Initialize the Flask app for testing
        self.app = create_app()
        self.client = self.app.test_client()

    def test_home_route(self):
        # Test the home route
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Video Authenticity Detector Backend is Running', response.data)

    def test_predict_no_file(self):
        # Test the predict route with no file
        response = self.client.post('/predict')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No file uploaded', response.data)

if __name__ == '__main__':
    unittest.main()