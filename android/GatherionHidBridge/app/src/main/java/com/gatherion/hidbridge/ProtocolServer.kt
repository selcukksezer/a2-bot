package com.gatherion.hidbridge

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Line protocol twin of Gatherion SerialHidAdapter / NetworkHidAdapter.
 */
class ProtocolServer(
    private val port: Int = 8765,
    private val onLine: (String) -> String?,
) {
    companion object {
        private const val TAG = "GatherionHidTcp"
    }

    private val running = AtomicBoolean(false)
    private var server: ServerSocket? = null

    fun start() {
        if (!running.compareAndSet(false, true)) return
        thread(name = "GatherionHidTcp", isDaemon = true) {
            try {
                val ss = ServerSocket(port)
                server = ss
                Log.i(TAG, "listening on $port")
                while (running.get()) {
                    val client = try {
                        ss.accept()
                    } catch (_: Exception) {
                        break
                    }
                    thread(name = "GatherionHidClient", isDaemon = true) {
                        handleClient(client)
                    }
                }
            } catch (exc: Exception) {
                Log.e(TAG, "server failed: $exc")
            } finally {
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try {
            server?.close()
        } catch (_: Exception) {
        }
        server = null
    }

    private fun handleClient(socket: Socket) {
        Log.i(TAG, "client ${socket.inetAddress.hostAddress}")
        try {
            socket.soTimeout = 0
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
            val out = socket.getOutputStream()
            while (running.get()) {
                val line = reader.readLine() ?: break
                val trimmed = line.trim()
                if (trimmed.isEmpty()) continue
                val reply = onLine(trimmed)
                if (reply != null) {
                    out.write((reply + "\n").toByteArray(Charsets.UTF_8))
                    out.flush()
                }
            }
        } catch (exc: Exception) {
            Log.d(TAG, "client closed: $exc")
        } finally {
            try {
                socket.close()
            } catch (_: Exception) {
            }
        }
    }
}
