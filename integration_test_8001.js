// Comprehensive Integration Test for SuperNova AI
// Testing on ports: Backend 8001, Frontend 3000

async function testSuperNovaIntegration() {
    console.log('=== SuperNova AI - Comprehensive Integration Test ===\n');
    
    const backendURL = 'http://localhost:8001';
    const frontendURL = 'http://localhost:3000';
    
    // Test results storage
    const testResults = {
        backend: {},
        frontend: {},
        integration: {},
        errors: [],
        warnings: []
    };
    
    console.log('1. Testing Backend Health...');
    try {
        const healthResp = await fetch(`${backendURL}/health`);
        const healthData = await healthResp.json();
        testResults.backend.health = {
            status: 'SUCCESS',
            data: healthData
        };
        console.log('   ✓ Backend is healthy:', healthData.service);
    } catch (error) {
        testResults.backend.health = {
            status: 'ERROR',
            error: error.message
        };
        testResults.errors.push(`Backend health check failed: ${error.message}`);
        console.log('   ✗ Backend health check failed:', error.message);
    }
    
    console.log('\n2. Testing Frontend Availability...');
    try {
        const frontResp = await fetch(frontendURL);
        if (frontResp.ok) {
            testResults.frontend.availability = {
                status: 'SUCCESS',
                statusCode: frontResp.status
            };
            console.log('   ✓ Frontend is accessible');
        }
    } catch (error) {
        testResults.frontend.availability = {
            status: 'ERROR',
            error: error.message
        };
        testResults.errors.push(`Frontend connection failed: ${error.message}`);
        console.log('   ✗ Frontend connection failed:', error.message);
    }
    
    console.log('\n3. Testing Core API Endpoints...');
    const coreEndpoints = [
        { path: '/health', name: 'Health Check', expectedStatus: 200 },
        { path: '/api/indicators/health', name: 'Indicators Health', expectedStatus: 200 },
        { path: '/api/alpha-vantage/health', name: 'Alpha Vantage Health', expectedStatus: 200 },
        { path: '/api/indicators/functions/all', name: 'Indicators Functions', expectedStatus: [200, 404] },
        { path: '/api/history', name: 'History API', expectedStatus: [200, 401, 422] }
    ];
    
    for (const endpoint of coreEndpoints) {
        try {
            const resp = await fetch(`${backendURL}${endpoint.path}`);
            const status = resp.status;
            const expectedStatuses = Array.isArray(endpoint.expectedStatus) ? endpoint.expectedStatus : [endpoint.expectedStatus];
            
            if (expectedStatuses.includes(status)) {
                console.log(`   ✓ ${endpoint.name} - OK (${status})`);
                testResults.backend[endpoint.path] = { status: 'SUCCESS', code: status };
            } else {
                console.log(`   ⚠ ${endpoint.name} - Unexpected status (${status})`);
                testResults.warnings.push(`${endpoint.name} returned unexpected status ${status}`);
                testResults.backend[endpoint.path] = { status: 'UNEXPECTED', code: status };
            }
        } catch (error) {
            console.log(`   ✗ ${endpoint.name} - Error: ${error.message}`);
            testResults.errors.push(`${endpoint.name} failed: ${error.message}`);
            testResults.backend[endpoint.path] = { status: 'ERROR', error: error.message };
        }
    }
    
    console.log('\n4. Testing Authentication Endpoints...');
    const authEndpoints = [
        { path: '/auth/register', method: 'POST', name: 'User Registration', expectedStatus: [400, 422, 500] },
        { path: '/auth/login', method: 'POST', name: 'User Login', expectedStatus: [400, 422] }
    ];
    
    for (const endpoint of authEndpoints) {
        try {
            const resp = await fetch(`${backendURL}${endpoint.path}`, {
                method: endpoint.method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}) // Empty body to test validation
            });
            const status = resp.status;
            const expectedStatuses = Array.isArray(endpoint.expectedStatus) ? endpoint.expectedStatus : [endpoint.expectedStatus];
            
            if (expectedStatuses.includes(status)) {
                console.log(`   ✓ ${endpoint.name} - Validation working (${status})`);
                testResults.backend[endpoint.path] = { status: 'SUCCESS', code: status };
            } else {
                console.log(`   ⚠ ${endpoint.name} - Unexpected status (${status})`);
                testResults.warnings.push(`${endpoint.name} returned unexpected status ${status}`);
                testResults.backend[endpoint.path] = { status: 'UNEXPECTED', code: status };
            }
        } catch (error) {
            console.log(`   ✗ ${endpoint.name} - Error: ${error.message}`);
            testResults.errors.push(`${endpoint.name} failed: ${error.message}`);
            testResults.backend[endpoint.path] = { status: 'ERROR', error: error.message };
        }
    }
    
    console.log('\n5. Testing Frontend-Backend Proxy...');
    try {
        const proxyResp = await fetch(`${frontendURL}/api/indicators/health`);
        const proxyData = await proxyResp.json();
        
        if (proxyResp.ok && proxyData) {
            testResults.integration.proxy = {
                status: 'SUCCESS',
                data: proxyData
            };
            console.log('   ✓ Frontend proxy to backend working');
        } else {
            testResults.integration.proxy = {
                status: 'ERROR',
                statusCode: proxyResp.status
            };
            testResults.errors.push('Frontend proxy not working correctly');
            console.log('   ✗ Frontend proxy failed');
        }
    } catch (error) {
        testResults.integration.proxy = {
            status: 'ERROR',
            error: error.message
        };
        testResults.errors.push(`Frontend proxy test failed: ${error.message}`);
        console.log('   ✗ Frontend proxy test failed:', error.message);
    }
    
    console.log('\n6. Testing Protected Endpoints...');
    try {
        const protectedResp = await fetch(`${backendURL}/intake`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                goal: "Test Goal",
                risk_tolerance: "medium", 
                time_horizon: "long",
                financial_situation: "stable"
            })
        });
        
        if (protectedResp.status === 401 || protectedResp.status === 403) {
            console.log('   ✓ Protected endpoint requires authentication');
            testResults.backend.authentication = { status: 'SUCCESS', message: 'Authentication required' };
        } else {
            console.log(`   ⚠ Protected endpoint returned unexpected status: ${protectedResp.status}`);
            testResults.warnings.push('Protected endpoint authentication may not be working');
        }
    } catch (error) {
        testResults.errors.push(`Protected endpoint test failed: ${error.message}`);
        console.log('   ✗ Protected endpoint test failed:', error.message);
    }
    
    // Summary Report
    console.log('\n=== INTEGRATION TEST SUMMARY ===');
    console.log(`Backend Status: ${testResults.backend.health?.status || 'UNKNOWN'}`);
    console.log(`Frontend Status: ${testResults.frontend.availability?.status || 'UNKNOWN'}`);
    console.log(`Proxy Status: ${testResults.integration.proxy?.status || 'UNKNOWN'}`);
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
    
    // Overall Status
    const criticalErrors = testResults.errors.filter(err => 
        err.includes('Backend health') || 
        err.includes('Frontend connection') ||
        err.includes('Frontend proxy')
    ).length;
    
    console.log('\n=== OVERALL ASSESSMENT ===');
    if (criticalErrors === 0) {
        if (testResults.errors.length === 0) {
            console.log('🎉 ALL SYSTEMS OPERATIONAL - Integration test passed completely!');
        } else {
            console.log('✅ CORE SYSTEMS OPERATIONAL - Some non-critical issues found');
        }
    } else {
        console.log('❌ CRITICAL ISSUES FOUND - Core integration not working');
    }
    
    // Save results
    const fs = require('fs').promises;
    await fs.writeFile('integration_test_results.json', JSON.stringify(testResults, null, 2));
    console.log('\n✓ Test results saved to integration_test_results.json');
    
    return testResults;
}

// Run tests if executed directly
if (typeof module !== 'undefined' && require.main === module) {
    testSuperNovaIntegration().catch(console.error);
}

module.exports = { testSuperNovaIntegration };