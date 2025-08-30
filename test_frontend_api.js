// Test script to check frontend and API interactions

async function testSuperNovaApp() {
    console.log('=== SuperNova AI - Comprehensive Test Report ===\n');
    
    const baseURL = 'http://localhost:8000';
    const frontendURL = 'http://localhost:3000';
    
    // Test results storage
    const testResults = {
        backend: {},
        frontend: {},
        errors: [],
        warnings: []
    };
    
    // 1. Test Backend Health
    console.log('1. Testing Backend Health...');
    try {
        const healthResp = await fetch(`${baseURL}/health`);
        const healthData = await healthResp.json();
        testResults.backend.health = {
            status: 'SUCCESS',
            data: healthData
        };
        console.log('   ✓ Backend is healthy');
    } catch (error) {
        testResults.backend.health = {
            status: 'ERROR',
            error: error.message
        };
        testResults.errors.push(`Backend health check failed: ${error.message}`);
        console.log('   ✗ Backend health check failed:', error.message);
    }
    
    // 2. Test Frontend Availability
    console.log('\n2. Testing Frontend Availability...');
    try {
        const frontResp = await fetch(frontendURL);
        if (frontResp.ok) {
            testResults.frontend.availability = {
                status: 'SUCCESS',
                statusCode: frontResp.status
            };
            console.log('   ✓ Frontend is accessible');
        } else {
            testResults.frontend.availability = {
                status: 'ERROR',
                statusCode: frontResp.status
            };
            testResults.errors.push(`Frontend returned status ${frontResp.status}`);
            console.log('   ✗ Frontend returned status:', frontResp.status);
        }
    } catch (error) {
        testResults.frontend.availability = {
            status: 'ERROR',
            error: error.message
        };
        testResults.errors.push(`Frontend connection failed: ${error.message}`);
        console.log('   ✗ Frontend connection failed:', error.message);
    }
    
    // 3. Test API Endpoints
    console.log('\n3. Testing API Endpoints...');
    const apiEndpoints = [
        '/api/indicators/health',
        '/api/indicators/functions/all',
        '/api/history',
        '/api/alpha-vantage/health'
    ];
    
    for (const endpoint of apiEndpoints) {
        try {
            const resp = await fetch(`${baseURL}${endpoint}`);
            const status = resp.status;
            
            if (status === 200) {
                console.log(`   ✓ ${endpoint} - OK (${status})`);
                testResults.backend[endpoint] = { status: 'SUCCESS', code: status };
            } else if (status === 401 || status === 403) {
                console.log(`   ⚠ ${endpoint} - Requires Authentication (${status})`);
                testResults.warnings.push(`${endpoint} requires authentication`);
                testResults.backend[endpoint] = { status: 'AUTH_REQUIRED', code: status };
            } else {
                console.log(`   ✗ ${endpoint} - Error (${status})`);
                testResults.errors.push(`${endpoint} returned status ${status}`);
                testResults.backend[endpoint] = { status: 'ERROR', code: status };
            }
        } catch (error) {
            console.log(`   ✗ ${endpoint} - Connection Error`);
            testResults.errors.push(`${endpoint} connection failed: ${error.message}`);
            testResults.backend[endpoint] = { status: 'CONNECTION_ERROR', error: error.message };
        }
    }
    
    // 4. Test WebSocket Connection
    console.log('\n4. Testing WebSocket Connection...');
    try {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        await new Promise((resolve, reject) => {
            ws.onopen = () => {
                console.log('   ✓ WebSocket connection established');
                testResults.backend.websocket = { status: 'SUCCESS' };
                ws.close();
                resolve();
            };
            
            ws.onerror = (error) => {
                console.log('   ✗ WebSocket connection failed');
                testResults.backend.websocket = { status: 'ERROR', error: 'Connection failed' };
                testResults.errors.push('WebSocket connection failed');
                reject(error);
            };
            
            setTimeout(() => {
                console.log('   ⚠ WebSocket connection timeout');
                testResults.backend.websocket = { status: 'TIMEOUT' };
                testResults.warnings.push('WebSocket connection timeout');
                ws.close();
                resolve();
            }, 5000);
        });
    } catch (error) {
        console.log('   ✗ WebSocket test error:', error.message);
        testResults.errors.push(`WebSocket test error: ${error.message}`);
    }
    
    // 5. Summary Report
    console.log('\n=== TEST SUMMARY ===');
    console.log(`Total Errors: ${testResults.errors.length}`);
    console.log(`Total Warnings: ${testResults.warnings.length}`);
    
    if (testResults.errors.length > 0) {
        console.log('\nErrors Found:');
        testResults.errors.forEach((err, i) => {
            console.log(`  ${i + 1}. ${err}`);
        });
    }
    
    if (testResults.warnings.length > 0) {
        console.log('\nWarnings:');
        testResults.warnings.forEach((warn, i) => {
            console.log(`  ${i + 1}. ${warn}`);
        });
    }
    
    // Save results to file
    const fs = require('fs').promises;
    await fs.writeFile('test_results.json', JSON.stringify(testResults, null, 2));
    console.log('\n✓ Test results saved to test_results.json');
    
    return testResults;
}

// Run tests if executed directly
if (typeof module !== 'undefined' && require.main === module) {
    testSuperNovaApp().catch(console.error);
}

module.exports = { testSuperNovaApp };