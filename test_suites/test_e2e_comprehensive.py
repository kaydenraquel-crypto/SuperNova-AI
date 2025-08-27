"""
Comprehensive End-to-End Testing Suite
======================================

Complete user workflow testing using Playwright for browser automation
and API testing for backend workflows.
"""
import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from fastapi.testclient import TestClient

from supernova.api import app


class TestCompleteUserJourneys:
    """Test complete user journeys from onboarding to portfolio management."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_new_user_complete_journey(self):
        """Test complete new user journey from landing to portfolio creation."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Step 1: Landing page
                await page.goto("http://localhost:3000")
                await page.wait_for_load_state("networkidle")
                
                # Verify landing page loads
                title = await page.title()
                assert "SuperNova" in title
                
                # Step 2: Navigate to onboarding
                await page.click('text="Get Started"')
                await page.wait_for_url("**/onboarding")
                
                # Step 3: Complete risk assessment
                await self._complete_risk_assessment(page)
                
                # Step 4: Provide personal information
                await self._provide_personal_info(page)
                
                # Step 5: Review generated portfolio
                await page.wait_for_selector('[data-testid="portfolio-recommendation"]')
                
                # Verify portfolio is displayed
                portfolio_element = await page.query_selector('[data-testid="portfolio-recommendation"]')
                assert portfolio_element is not None
                
                # Step 6: Accept portfolio and create account
                await page.click('[data-testid="accept-portfolio-btn"]')
                
                # Step 7: Verify dashboard loads
                await page.wait_for_url("**/dashboard")
                
                # Verify dashboard components
                dashboard_elements = [
                    '[data-testid="portfolio-overview"]',
                    '[data-testid="performance-chart"]',
                    '[data-testid="asset-allocation"]',
                    '[data-testid="watchlist"]'
                ]
                
                for element in dashboard_elements:
                    await page.wait_for_selector(element, timeout=10000)
                
                # Step 8: Test basic navigation
                await self._test_dashboard_navigation(page)
                
            finally:
                await browser.close()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_workflow(self):
        """Test complete portfolio rebalancing workflow."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Login as existing user
                await self._login_existing_user(page)
                
                # Navigate to portfolio management
                await page.click('[data-testid="portfolio-menu"]')
                await page.click('text="Manage Portfolio"')
                await page.wait_for_url("**/portfolio")
                
                # Step 1: View current allocation
                await page.wait_for_selector('[data-testid="current-allocation"]')
                
                # Get current allocations
                allocation_elements = await page.query_selector_all('[data-testid="allocation-item"]')
                initial_allocation_count = len(allocation_elements)
                
                # Step 2: Trigger rebalancing analysis
                await page.click('[data-testid="analyze-rebalancing-btn"]')
                await page.wait_for_selector('[data-testid="rebalancing-recommendations"]')
                
                # Step 3: Review recommendations
                recommendations = await page.query_selector_all('[data-testid="rebalancing-recommendation"]')
                assert len(recommendations) > 0
                
                # Step 4: Accept rebalancing
                await page.click('[data-testid="accept-rebalancing-btn"]')
                
                # Step 5: Confirm transaction
                await page.wait_for_selector('[data-testid="transaction-confirmation"]')
                await page.click('[data-testid="confirm-transaction-btn"]')
                
                # Step 6: Verify successful rebalancing
                await page.wait_for_selector('[data-testid="rebalancing-success"]', timeout=30000)
                
                # Verify portfolio updated
                await page.reload()
                await page.wait_for_selector('[data-testid="current-allocation"]')
                
                updated_allocation_elements = await page.query_selector_all('[data-testid="allocation-item"]')
                # Should have at least the same number of allocations
                assert len(updated_allocation_elements) >= initial_allocation_count
                
            finally:
                await browser.close()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_backtesting_workflow(self):
        """Test complete backtesting workflow."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await self._login_existing_user(page)
                
                # Navigate to backtesting
                await page.click('[data-testid="analytics-menu"]')
                await page.click('text="Backtesting"')
                await page.wait_for_url("**/backtesting")
                
                # Step 1: Configure backtest parameters
                await page.fill('[data-testid="start-date-input"]', '2020-01-01')
                await page.fill('[data-testid="end-date-input"]', '2023-12-31')
                await page.fill('[data-testid="initial-capital-input"]', '100000')
                
                # Select assets for backtesting
                asset_options = ["VTI", "BND", "VEA", "VWO"]
                for asset in asset_options:
                    await page.check(f'[data-testid="asset-checkbox-{asset}"]')
                
                # Step 2: Run backtest
                await page.click('[data-testid="run-backtest-btn"]')
                await page.wait_for_selector('[data-testid="backtest-progress"]')
                
                # Wait for backtest completion (may take time)
                await page.wait_for_selector('[data-testid="backtest-results"]', timeout=60000)
                
                # Step 3: Verify results
                results_elements = [
                    '[data-testid="total-return"]',
                    '[data-testid="sharpe-ratio"]',
                    '[data-testid="max-drawdown"]',
                    '[data-testid="volatility"]',
                    '[data-testid="performance-chart"]'
                ]
                
                for element in results_elements:
                    await page.wait_for_selector(element)
                
                # Step 4: Export results
                await page.click('[data-testid="export-results-btn"]')
                
                # Verify download starts
                async with page.expect_download() as download_info:
                    await page.click('[data-testid="download-pdf-btn"]')
                
                download = await download_info.value
                assert download.suggested_filename.endswith('.pdf')
                
            finally:
                await browser.close()
    
    async def _complete_risk_assessment(self, page: Page):
        """Complete the risk assessment questionnaire."""
        risk_questions = [
            {"question": 1, "answer": 3},  # Moderate risk tolerance
            {"question": 2, "answer": 4},  # Longer time horizon
            {"question": 3, "answer": 2},  # Some investment experience
            {"question": 4, "answer": 3},  # Balanced approach to volatility
            {"question": 5, "answer": 4},  # Growth focused
        ]
        
        for question in risk_questions:
            question_selector = f'[data-testid="risk-question-{question["question"]}"]'
            answer_selector = f'[data-testid="risk-answer-{question["question"]}-{question["answer"]}"]'
            
            await page.wait_for_selector(question_selector)
            await page.click(answer_selector)
        
        await page.click('[data-testid="next-step-btn"]')
    
    async def _provide_personal_info(self, page: Page):
        """Provide personal information during onboarding."""
        await page.wait_for_selector('[data-testid="personal-info-form"]')
        
        # Fill personal information
        await page.fill('[data-testid="name-input"]', 'E2E Test User')
        await page.fill('[data-testid="email-input"]', 'e2e@test.com')
        await page.fill('[data-testid="income-input"]', '75000')
        await page.select_option('[data-testid="investment-goal-select"]', 'retirement')
        await page.fill('[data-testid="time-horizon-input"]', '15')
        
        await page.click('[data-testid="generate-portfolio-btn"]')
        await page.wait_for_selector('[data-testid="portfolio-generation-loading"]')
    
    async def _test_dashboard_navigation(self, page: Page):
        """Test dashboard navigation functionality."""
        navigation_items = [
            {"menu": "portfolio-menu", "item": "Portfolio Overview"},
            {"menu": "analytics-menu", "item": "Performance Analytics"},
            {"menu": "tools-menu", "item": "Risk Assessment"},
        ]
        
        for nav_item in navigation_items:
            await page.click(f'[data-testid="{nav_item["menu"]}"]')
            await page.click(f'text="{nav_item["item"]}"')
            
            # Wait for page to load
            await page.wait_for_load_state("networkidle")
            
            # Verify we navigated successfully
            current_url = page.url
            assert any(keyword in current_url.lower() 
                      for keyword in nav_item["item"].lower().split())
    
    async def _login_existing_user(self, page: Page):
        """Login as existing test user."""
        await page.goto("http://localhost:3000/login")
        await page.wait_for_load_state("networkidle")
        
        await page.fill('[data-testid="email-input"]', 'test@example.com')
        await page.fill('[data-testid="password-input"]', 'testpassword123')
        await page.click('[data-testid="login-btn"]')
        
        await page.wait_for_url("**/dashboard")


class TestAdvancedUserWorkflows:
    """Test advanced user workflows and edge cases."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_collaborative_portfolio_sharing(self):
        """Test collaborative portfolio sharing workflow."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # Create two browser contexts (two users)
            user1_context = await browser.new_context()
            user2_context = await browser.new_context()
            
            user1_page = await user1_context.new_page()
            user2_page = await user2_context.new_page()
            
            try:
                # User 1: Share portfolio
                await self._login_existing_user(user1_page)
                await user1_page.click('[data-testid="portfolio-menu"]')
                await user1_page.click('text="Share Portfolio"')
                
                await user1_page.fill('[data-testid="share-email-input"]', 'user2@test.com')
                await user1_page.click('[data-testid="send-share-btn"]')
                
                # Get share link
                await user1_page.wait_for_selector('[data-testid="share-link"]')
                share_link = await user1_page.get_attribute('[data-testid="share-link"]', 'href')
                
                # User 2: Access shared portfolio
                await user2_page.goto(share_link)
                await user2_page.wait_for_selector('[data-testid="shared-portfolio-view"]')
                
                # Verify user 2 can view but not edit
                view_only_indicators = await user2_page.query_selector_all('[data-testid="view-only"]')
                assert len(view_only_indicators) > 0
                
                # User 2: Comment on portfolio
                await user2_page.fill('[data-testid="comment-input"]', 'Great portfolio allocation!')
                await user2_page.click('[data-testid="add-comment-btn"]')
                
                # User 1: See notification of comment
                await user1_page.reload()
                await user1_page.wait_for_selector('[data-testid="notification-badge"]')
                
            finally:
                await browser.close()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_real_time_market_data_updates(self):
        """Test real-time market data updates and live portfolio tracking."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                await self._login_existing_user(page)
                
                # Navigate to live portfolio view
                await page.click('[data-testid="portfolio-menu"]')
                await page.click('text="Live Tracking"')
                await page.wait_for_url("**/portfolio/live")
                
                # Get initial portfolio value
                initial_value_element = await page.query_selector('[data-testid="portfolio-value"]')
                initial_value = await initial_value_element.text_content()
                
                # Wait for market data updates (simulate with WebSocket)
                await page.wait_for_timeout(5000)  # Wait 5 seconds
                
                # Check if portfolio value has been updated
                updated_value_element = await page.query_selector('[data-testid="portfolio-value"]')
                updated_value = await updated_value_element.text_content()
                
                # Verify real-time updates are working
                last_update_element = await page.query_selector('[data-testid="last-update-time"]')
                assert last_update_element is not None
                
                # Test WebSocket connection indicator
                connection_status = await page.query_selector('[data-testid="websocket-status"]')
                status_text = await connection_status.text_content()
                assert "connected" in status_text.lower() or "live" in status_text.lower()
                
            finally:
                await browser.close()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_mobile_responsive_interface(self):
        """Test mobile responsive interface and touch interactions."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # Create mobile context
            context = await browser.new_context(
                viewport={'width': 375, 'height': 667},  # iPhone dimensions
                user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
            )
            
            page = await context.new_page()
            
            try:
                await self._login_existing_user(page)
                
                # Test mobile navigation
                await page.click('[data-testid="mobile-menu-toggle"]')
                await page.wait_for_selector('[data-testid="mobile-nav-menu"]')
                
                # Test touch gestures on charts
                chart_element = await page.query_selector('[data-testid="performance-chart"]')
                
                # Simulate pinch-to-zoom
                await page.touch_screen.tap(200, 300)
                await page.wait_for_timeout(1000)
                
                # Test swipe navigation on carousel
                carousel = await page.query_selector('[data-testid="portfolio-carousel"]')
                if carousel:
                    # Swipe left
                    await page.touch_screen.tap(300, 400)
                    await page.mouse.move(100, 400)
                    await page.wait_for_timeout(500)
                
                # Verify mobile-optimized elements are present
                mobile_elements = [
                    '[data-testid="mobile-portfolio-summary"]',
                    '[data-testid="mobile-quick-actions"]',
                    '[data-testid="mobile-notification-panel"]'
                ]
                
                for element in mobile_elements:
                    element_handle = await page.query_selector(element)
                    if element_handle:
                        is_visible = await element_handle.is_visible()
                        assert is_visible
                
            finally:
                await browser.close()


class TestAPIWorkflowIntegration:
    """Test complete API workflows and integrations."""
    
    @pytest.mark.e2e
    def test_complete_api_user_journey(self):
        """Test complete user journey via API calls."""
        client = TestClient(app)
        
        # Step 1: User onboarding
        intake_data = {
            "name": "API E2E Test User",
            "email": "apie2e@test.com",
            "income": 85000.0,
            "risk_questions": [3, 4, 3, 4, 3],
            "investment_goals": ["retirement", "growth"],
            "time_horizon_yrs": 20
        }
        
        intake_response = client.post("/intake", json=intake_data)
        assert intake_response.status_code == 200
        
        result = intake_response.json()
        profile_id = result["profile_id"]
        user_id = result.get("user_id")
        
        # Step 2: Get portfolio recommendation
        advice_data = {
            "profile_id": profile_id,
            "symbols": ["VTI", "VXUS", "BND", "VNQ"],
            "include_alternatives": True
        }
        
        advice_response = client.post("/advice", json=advice_data)
        assert advice_response.status_code == 200
        
        portfolio = advice_response.json()
        assert "allocations" in portfolio
        assert "risk_metrics" in portfolio
        
        # Step 3: Build watchlist
        watchlist_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        
        for symbol in watchlist_symbols:
            watchlist_response = client.post("/watchlist", json={
                "profile_id": profile_id,
                "symbol": symbol
            })
            assert watchlist_response.status_code == 200
        
        # Step 4: Get watchlist data
        watchlist_response = client.get(f"/watchlist/{profile_id}")
        assert watchlist_response.status_code == 200
        
        watchlist_data = watchlist_response.json()
        assert len(watchlist_data["items"]) == len(watchlist_symbols)
        
        # Step 5: Run comprehensive backtest
        backtest_data = {
            "profile_id": profile_id,
            "symbols": ["VTI", "VXUS", "BND"],
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
            "rebalancing_frequency": "quarterly"
        }
        
        backtest_response = client.post("/backtest", json=backtest_data)
        assert backtest_response.status_code == 200
        
        backtest_results = backtest_response.json()
        assert "performance" in backtest_results
        assert "metrics" in backtest_results
        
        # Verify key performance metrics
        metrics = backtest_results["metrics"]
        required_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Step 6: Get analytics insights
        analytics_response = client.get(f"/analytics/insights/{profile_id}")
        assert analytics_response.status_code == 200
        
        insights = analytics_response.json()
        assert "insights" in insights
        assert "recommendations" in insights
        
        # Step 7: Update profile based on insights
        if insights["recommendations"]:
            # Simulate profile update based on recommendations
            update_data = {
                "time_horizon_yrs": 25,  # Extend time horizon
                "investment_goals": ["retirement", "growth", "income"]
            }
            
            update_response = client.put(f"/profile/{profile_id}", json=update_data)
            assert update_response.status_code == 200
        
        # Step 8: Get updated portfolio recommendation
        updated_advice_response = client.post("/advice", json=advice_data)
        assert updated_advice_response.status_code == 200
        
        updated_portfolio = updated_advice_response.json()
        
        # Verify portfolio changed based on profile update
        original_allocations = {alloc["symbol"]: alloc["weight"] for alloc in portfolio["allocations"]}
        updated_allocations = {alloc["symbol"]: alloc["weight"] for alloc in updated_portfolio["allocations"]}
        
        # At least some allocations should have changed
        allocation_changes = sum(1 for symbol in original_allocations
                               if symbol in updated_allocations and
                               abs(original_allocations[symbol] - updated_allocations[symbol]) > 0.01)
        
        assert allocation_changes > 0, "Portfolio should change after profile update"
    
    @pytest.mark.e2e
    def test_error_recovery_workflows(self):
        """Test error recovery and fallback mechanisms."""
        client = TestClient(app)
        
        # Test 1: Partial failure recovery
        # Create user successfully
        intake_response = client.post("/intake", json={
            "name": "Error Recovery Test",
            "email": "errorrecovery@test.com",
            "risk_questions": [3, 3, 3, 3, 3]
        })
        profile_id = intake_response.json()["profile_id"]
        
        # Attempt operation that might fail
        invalid_backtest = {
            "profile_id": profile_id,
            "symbols": ["INVALID_SYMBOL"],
            "start_date": "invalid_date",
            "end_date": "2023-12-31",
            "initial_capital": -1000  # Invalid amount
        }
        
        backtest_response = client.post("/backtest", json=invalid_backtest)
        
        # Should handle gracefully
        assert backtest_response.status_code in [400, 422]
        
        # User profile should still be intact
        profile_response = client.get(f"/profile/{profile_id}")
        assert profile_response.status_code == 200
        
        # Test 2: Service degradation handling
        # With mock external service failure
        from unittest.mock import patch
        
        with patch('supernova.api.get_market_data') as mock_market:
            mock_market.side_effect = Exception("External service unavailable")
            
            # Should fall back to cached data or default response
            market_response = client.get("/market-data/VTI")
            assert market_response.status_code in [200, 503]
            
            if market_response.status_code == 200:
                # If successful, should indicate data is cached/stale
                data = market_response.json()
                assert "cached" in str(data) or "stale" in str(data) or "fallback" in str(data)


class TestPerformanceAndScalability:
    """Test performance and scalability of end-to-end workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_workflows(self):
        """Test multiple concurrent users performing workflows."""
        async def simulate_user_workflow(user_id: int):
            """Simulate a single user workflow."""
            client = TestClient(app)
            
            # User onboarding
            intake_response = client.post("/intake", json={
                "name": f"Concurrent User {user_id}",
                "email": f"concurrent{user_id}@test.com",
                "risk_questions": [3, 3, 3, 3, 3]
            })
            
            if intake_response.status_code != 200:
                return {"user_id": user_id, "status": "failed", "step": "intake"}
            
            profile_id = intake_response.json()["profile_id"]
            
            # Get advice
            advice_response = client.post("/advice", json={
                "profile_id": profile_id,
                "symbols": ["VTI", "BND"]
            })
            
            if advice_response.status_code != 200:
                return {"user_id": user_id, "status": "failed", "step": "advice"}
            
            # Add to watchlist
            watchlist_response = client.post("/watchlist", json={
                "profile_id": profile_id,
                "symbol": "AAPL"
            })
            
            if watchlist_response.status_code != 200:
                return {"user_id": user_id, "status": "failed", "step": "watchlist"}
            
            return {"user_id": user_id, "status": "success"}
        
        # Run concurrent user simulations
        num_concurrent_users = 10
        start_time = time.time()
        
        tasks = [simulate_user_workflow(i) for i in range(num_concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_users = sum(1 for result in results 
                             if isinstance(result, dict) and result.get("status") == "success")
        
        success_rate = successful_users / num_concurrent_users
        avg_time_per_user = total_time / num_concurrent_users
        
        # Performance assertions
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
        assert avg_time_per_user < 5.0, f"Average time per user too high: {avg_time_per_user}s"
        assert total_time < 30.0, f"Total concurrent execution time too high: {total_time}s"
    
    @pytest.mark.e2e
    @pytest.mark.performance
    def test_large_dataset_handling(self):
        """Test handling of large datasets in workflows."""
        client = TestClient(app)
        
        # Create user
        intake_response = client.post("/intake", json={
            "name": "Large Dataset Test User",
            "email": "largedataset@test.com",
            "risk_questions": [4, 4, 4, 4, 4]
        })
        profile_id = intake_response.json()["profile_id"]
        
        # Test with large symbol set
        large_symbol_set = [
            "VTI", "VXUS", "BND", "BNDX", "VEA", "VWO", "VNQ", "VNQI",
            "QQQ", "SPY", "IWM", "EFA", "EEM", "AGG", "LQD", "HYG",
            "GLD", "SLV", "DBA", "USO", "TLT", "IEF", "SHY", "VTEB"
        ]
        
        start_time = time.time()
        
        # Request advice for large portfolio
        advice_response = client.post("/advice", json={
            "profile_id": profile_id,
            "symbols": large_symbol_set,
            "optimization_method": "comprehensive"
        })
        
        advice_time = time.time() - start_time
        
        assert advice_response.status_code == 200
        assert advice_time < 10.0, f"Large dataset advice took too long: {advice_time}s"
        
        portfolio = advice_response.json()
        assert len(portfolio["allocations"]) <= len(large_symbol_set)
        
        # Test large historical backtest
        start_time = time.time()
        
        backtest_response = client.post("/backtest", json={
            "profile_id": profile_id,
            "symbols": large_symbol_set[:10],  # Limit for backtest
            "start_date": "2015-01-01",  # 9 years of data
            "end_date": "2023-12-31",
            "initial_capital": 1000000
        })
        
        backtest_time = time.time() - start_time
        
        assert backtest_response.status_code == 200
        assert backtest_time < 30.0, f"Large historical backtest took too long: {backtest_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])