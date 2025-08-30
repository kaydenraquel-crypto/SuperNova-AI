// WebSocket Test for SuperNova AI
const WebSocket = require('ws');

async function testWebSocketConnections() {
    console.log('=== WebSocket Connection Test ===\n');
    
    const testResults = {
        socketio: null,
        websocket: null,
        errors: [],
        warnings: []
    };
    
    // Test Socket.IO endpoint
    console.log('1. Testing Socket.IO connection...');
    try {
        const socketioUrl = 'ws://localhost:8001/socket.io/?EIO=4&transport=websocket';
        const ws1 = new WebSocket(socketioUrl);
        
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                ws1.close();
                testResults.socketio = { status: 'TIMEOUT' };
                testResults.warnings.push('Socket.IO connection timeout');
                console.log('   ⚠ Socket.IO connection timeout');
                resolve();
            }, 5000);
            
            ws1.onopen = () => {
                clearTimeout(timeout);
                console.log('   ✓ Socket.IO connection established');
                testResults.socketio = { status: 'SUCCESS' };
                ws1.close();
                resolve();
            };
            
            ws1.onerror = (error) => {
                clearTimeout(timeout);
                console.log('   ✗ Socket.IO connection failed');
                testResults.socketio = { status: 'ERROR', error: 'Connection failed' };
                testResults.errors.push('Socket.IO connection failed');
                ws1.close();
                resolve();
            };
            
            ws1.onclose = (event) => {
                if (event.code !== 1000) { // 1000 is normal closure
                    console.log(`   ⚠ Socket.IO closed with code: ${event.code}`);
                }
            };
        });
    } catch (error) {
        console.log('   ✗ Socket.IO test error:', error.message);
        testResults.socketio = { status: 'ERROR', error: error.message };
        testResults.errors.push(`Socket.IO test error: ${error.message}`);
    }
    
    // Test regular WebSocket endpoint
    console.log('\n2. Testing WebSocket endpoint...');
    try {
        const wsUrl = 'ws://localhost:8001/ws';
        const ws2 = new WebSocket(wsUrl);
        
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                ws2.close();
                testResults.websocket = { status: 'TIMEOUT' };
                testResults.warnings.push('WebSocket connection timeout');
                console.log('   ⚠ WebSocket connection timeout');
                resolve();
            }, 5000);
            
            ws2.onopen = () => {
                clearTimeout(timeout);
                console.log('   ✓ WebSocket connection established');
                testResults.websocket = { status: 'SUCCESS' };
                
                // Test sending a message
                ws2.send(JSON.stringify({ type: 'test', message: 'hello' }));
                
                setTimeout(() => {
                    ws2.close();
                    resolve();
                }, 1000);
            };
            
            ws2.onerror = (error) => {
                clearTimeout(timeout);
                console.log('   ✗ WebSocket connection failed');
                testResults.websocket = { status: 'ERROR', error: 'Connection failed' };
                testResults.errors.push('WebSocket connection failed');
                ws2.close();
                resolve();
            };
            
            ws2.onmessage = (event) => {
                console.log('   📨 Received message:', event.data);
            };
            
            ws2.onclose = (event) => {
                if (event.code !== 1000) { // 1000 is normal closure
                    console.log(`   ⚠ WebSocket closed with code: ${event.code}`);
                }
            };
        });
    } catch (error) {
        console.log('   ✗ WebSocket test error:', error.message);
        testResults.websocket = { status: 'ERROR', error: error.message };
        testResults.errors.push(`WebSocket test error: ${error.message}`);
    }
    
    // Summary
    console.log('\n=== WebSocket Test Summary ===');
    console.log(`Socket.IO Status: ${testResults.socketio?.status || 'NOT_TESTED'}`);
    console.log(`WebSocket Status: ${testResults.websocket?.status || 'NOT_TESTED'}`);
    console.log(`Total Errors: ${testResults.errors.length}`);
    console.log(`Total Warnings: ${testResults.warnings.length}`);
    
    if (testResults.errors.length > 0) {
        console.log('\nErrors:');
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
    
    // Overall assessment
    const workingConnections = [testResults.socketio, testResults.websocket].filter(r => r?.status === 'SUCCESS').length;
    console.log('\n=== WebSocket Assessment ===');
    if (workingConnections === 2) {
        console.log('🎉 All WebSocket connections working!');
    } else if (workingConnections === 1) {
        console.log('✅ Some WebSocket connections working');
    } else {
        console.log('❌ No WebSocket connections working');
    }
    
    return testResults;
}

// Run test if executed directly
if (require.main === module) {
    testWebSocketConnections().catch(console.error);
}

module.exports = { testWebSocketConnections };