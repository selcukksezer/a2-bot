package com.gatherion.hidbridge

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.text.format.Formatter
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.net.NetworkInterface

class MainActivity : AppCompatActivity() {
    private lateinit var statusView: TextView
    private lateinit var ipView: TextView
    private var service: HidBridgeService? = null

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            service = (binder as HidBridgeService.LocalBinder).getService()
            statusView.text = service?.statusText ?: "Service bound"
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            statusView.text = intent?.getStringExtra("status") ?: return
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        statusView = findViewById(R.id.status)
        ipView = findViewById(R.id.ipInfo)
        ipView.text = "Wi-Fi IP: ${localIpv4()}\nTCP port: ${HidBridgeService.TCP_PORT}"

        findViewById<Button>(R.id.btnStart).setOnClickListener {
            ensurePermissions {
                startBridge()
            }
        }
        findViewById<Button>(R.id.btnConnect).setOnClickListener {
            startService(Intent(this, HidBridgeService::class.java).setAction("connect"))
            service?.connectFirstBonded()
        }
        findViewById<Button>(R.id.btnStop).setOnClickListener {
            stopService(Intent(this, HidBridgeService::class.java))
            statusView.text = getString(R.string.status_idle)
        }
        findViewById<Button>(R.id.btnTestTab).setOnClickListener {
            startService(Intent(this, HidBridgeService::class.java).setAction("test_tab"))
            Toast.makeText(this, "Tab sent (if HID connected)", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onStart() {
        super.onStart()
        ContextCompat.registerReceiver(
            this,
            statusReceiver,
            IntentFilter(HidBridgeService.ACTION_STATUS),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )
        bindService(Intent(this, HidBridgeService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        try {
            unregisterReceiver(statusReceiver)
        } catch (_: Exception) {
        }
        try {
            unbindService(connection)
        } catch (_: Exception) {
        }
        super.onStop()
    }

    private fun startBridge() {
        val intent = Intent(this, HidBridgeService::class.java)
        ContextCompat.startForegroundService(this, intent)
        statusView.text = "Starting HID + TCP…"
        Toast.makeText(this, "Pair this phone as a keyboard in Windows Bluetooth", Toast.LENGTH_LONG).show()
    }

    private fun ensurePermissions(onOk: () -> Unit) {
        val need = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= 31) {
            listOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_ADVERTISE,
                Manifest.permission.BLUETOOTH_SCAN,
            ).forEach {
                if (ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                    need.add(it)
                }
            }
        }
        if (Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                need.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (need.isEmpty()) {
            onOk()
            return
        }
        ActivityCompat.requestPermissions(this, need.toTypedArray(), 42)
        // User must tap Start again after granting.
        Toast.makeText(this, "Grant Bluetooth, then tap Start again", Toast.LENGTH_LONG).show()
    }

    private fun localIpv4(): String {
        try {
            val wm = applicationContext.getSystemService(WIFI_SERVICE) as WifiManager
            @Suppress("DEPRECATION")
            val ip = Formatter.formatIpAddress(wm.connectionInfo.ipAddress)
            if (ip != null && ip != "0.0.0.0") return ip
        } catch (_: Exception) {
        }
        try {
            NetworkInterface.getNetworkInterfaces()?.toList()?.forEach { nif ->
                nif.inetAddresses.toList().forEach { addr ->
                    if (!addr.isLoopbackAddress && addr.hostAddress?.contains(':') == false) {
                        return addr.hostAddress ?: "—"
                    }
                }
            }
        } catch (_: Exception) {
        }
        return "—"
    }
}
