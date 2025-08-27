"""
Authentication System Integration Test
Tests the complete authentication flow with real API endpoints
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust as needed
TEST_USER_EMAIL = "test_auth_user@example.com"
TEST_USER_PASSWORD = "TestPassword123!"
TEST_USER_NAME = "Test Auth User"

class AuthenticationTester:
    """Test the authentication system end-to-end"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None
        self.refresh_token = None
        self.user_data = None
    
    def test_user_registration(self):
        """Test user registration"""
        print("Testing user registration...")
        
        registration_data = {
            "name": TEST_USER_NAME,
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD,
            "confirm_password": TEST_USER_PASSWORD,
            "role": "user"
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/register",
            json=registration_data
        )
        
        if response.status_code == 200:
            print("✅ Registration successful")
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.user_data = data["user"]
            print(f"   User ID: {self.user_data['id']}")
            print(f"   Email: {self.user_data['email']}")
            print(f"   Role: {self.user_data['role']}")
            return True
        elif response.status_code == 409:
            print("ℹ️  User already exists, will test login instead")
            return self.test_user_login()
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_user_login(self):
        """Test user login"""
        print("Testing user login...")
        
        login_data = {
            "email": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json=login_data
        )
        
        if response.status_code == 200:
            print("✅ Login successful")
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.user_data = data["user"]
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_authenticated_request(self):
        """Test making authenticated requests"""
        print("Testing authenticated requests...")
        
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = self.session.get(
            f"{self.base_url}/auth/profile",
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Authenticated request successful")
            profile = response.json()
            print(f"   Profile retrieved: {profile['name']} ({profile['email']})")
            return True
        else:
            print(f"❌ Authenticated request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_token_refresh(self):
        """Test token refresh"""
        print("Testing token refresh...")
        
        if not self.refresh_token:
            print("❌ No refresh token available")
            return False
        
        refresh_data = {
            "refresh_token": self.refresh_token
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/refresh",
            json=refresh_data
        )
        
        if response.status_code == 200:
            print("✅ Token refresh successful")
            data = response.json()
            old_token = self.access_token
            self.access_token = data["access_token"]
            print(f"   Token refreshed (changed: {old_token != self.access_token})")
            return True
        else:
            print(f"❌ Token refresh failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_protected_endpoint(self):
        """Test accessing a protected endpoint"""
        print("Testing protected endpoint access...")
        
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Test the intake endpoint (requires authentication)
        intake_data = {
            "name": "Test Profile",
            "email": TEST_USER_EMAIL,
            "time_horizon_yrs": 5,
            "objectives": "growth",
            "constraints": "low risk",
            "risk_questions": [3, 2, 4, 3, 2]
        }
        
        response = self.session.post(
            f"{self.base_url}/intake",
            json=intake_data,
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Protected endpoint access successful")
            data = response.json()
            print(f"   Profile created with ID: {data['profile_id']}")
            print(f"   Risk score: {data['risk_score']}")
            return True
        else:
            print(f"❌ Protected endpoint access failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without token"""
        print("Testing unauthorized access...")
        
        # Try to access protected endpoint without token
        intake_data = {
            "name": "Test Profile",
            "email": TEST_USER_EMAIL,
            "risk_questions": []
        }
        
        response = self.session.post(
            f"{self.base_url}/intake",
            json=intake_data
        )
        
        if response.status_code == 401:
            print("✅ Unauthorized access correctly blocked")
            return True
        else:
            print(f"❌ Unauthorized access not blocked: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def test_password_change(self):
        """Test password change functionality"""
        print("Testing password change...")
        
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        new_password = "NewTestPassword123!"
        
        password_data = {
            "current_password": TEST_USER_PASSWORD,
            "new_password": new_password,
            "confirm_password": new_password
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/password/change",
            json=password_data,
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Password change successful")
            
            # Change it back for other tests
            password_data = {
                "current_password": new_password,
                "new_password": TEST_USER_PASSWORD,
                "confirm_password": TEST_USER_PASSWORD
            }
            
            # Login again with new password to get new token
            self.test_user_login()
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            self.session.post(
                f"{self.base_url}/auth/password/change",
                json=password_data,
                headers=headers
            )
            print("   Password reverted for other tests")
            return True
        else:
            print(f"❌ Password change failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_mfa_setup(self):
        """Test MFA setup"""
        print("Testing MFA setup...")
        
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        mfa_data = {
            "password": TEST_USER_PASSWORD
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/mfa/setup",
            json=mfa_data,
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ MFA setup successful")
            data = response.json()
            print(f"   Secret provided: {len(data['secret'])} characters")
            print(f"   QR code provided: {len(data['qr_code'])} characters")
            print(f"   Backup codes provided: {len(data['backup_codes'])}")
            return True
        else:
            print(f"❌ MFA setup failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_password_reset_request(self):
        """Test password reset request"""
        print("Testing password reset request...")
        
        reset_data = {
            "email": TEST_USER_EMAIL
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/password/reset",
            json=reset_data
        )
        
        if response.status_code == 200:
            print("✅ Password reset request successful")
            data = response.json()
            print(f"   Message: {data['message']}")
            return True
        else:
            print(f"❌ Password reset request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def test_logout(self):
        """Test user logout"""
        print("Testing user logout...")
        
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        logout_data = {
            "all_devices": False
        }
        
        response = self.session.post(
            f"{self.base_url}/auth/logout",
            json=logout_data,
            headers=headers
        )
        
        if response.status_code == 200:
            print("✅ Logout successful")
            self.access_token = None
            self.refresh_token = None
            return True
        else:
            print(f"❌ Logout failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    
    def run_all_tests(self):
        """Run all authentication tests"""
        print("🚀 Starting Authentication System Integration Tests")
        print("=" * 60)
        
        tests = [
            self.test_user_registration,
            self.test_authenticated_request,
            self.test_token_refresh,
            self.test_unauthorized_access,
            self.test_protected_endpoint,
            self.test_password_change,
            self.test_mfa_setup,
            self.test_password_reset_request,
            self.test_logout
        ]
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(tests, 1):
            print(f"\n[Test {i}/{len(tests)}]")
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test failed with exception: {str(e)}")
                failed += 1
            
            time.sleep(0.5)  # Small delay between tests
        
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All authentication tests passed!")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        return failed == 0


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperNova Authentication System Integration Tests")
    parser.add_argument("--url", default=BASE_URL, help="Base URL for the API server")
    parser.add_argument("--quick", action="store_true", help="Run only essential tests")
    
    args = parser.parse_args()
    
    tester = AuthenticationTester(args.url)
    
    try:
        # Test if server is running
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  Server health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {args.url}")
        print(f"   Error: {str(e)}")
        print("   Make sure the SuperNova API server is running")
        return False
    
    if args.quick:
        print("Running quick authentication tests...")
        tests = [
            tester.test_user_registration,
            tester.test_authenticated_request,
            tester.test_logout
        ]
        
        for test in tests:
            if not test():
                return False
        print("✅ Quick tests completed successfully")
    else:
        return tester.run_all_tests()
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)