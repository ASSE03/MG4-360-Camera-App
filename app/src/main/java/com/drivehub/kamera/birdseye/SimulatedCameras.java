package com.drivehub.kamera.birdseye;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Calibration of the simulated camera rig used by the "sim" build (emulator testing).
 *
 * <p>KEEP IN SYNC with {@code app/src/main/cpp/sim_camera_source.h}: the native side renders its
 * parking scene through exactly these four fisheye cameras, so the bird's-eye and 3D views that
 * the app builds from this calibration line up with the pictures.
 *
 * <p>World frame of the vehicle calibration: millimetres, x right, y towards the rear, z down,
 * ground at z = 0, origin at the rear axle. Order: left, right, front, rear.
 */
public final class SimulatedCameras {
    private SimulatedCameras() {}

    private static final double FX = 230, FY = 230, CX = 360, CY = 240;
    private static final double[] K = {0.015, -0.004, 0.0005, 0};

    /** {centre xyz, forward xyz, image-right xyz} for left, right, front, rear. */
    private static double[][] rig() {
        double cs = Math.cos(Math.toRadians(65)), ss = Math.sin(Math.toRadians(65));
        double cf = Math.cos(Math.toRadians(40)), sf = Math.sin(Math.toRadians(40));
        double cr = Math.cos(Math.toRadians(45)), sr = Math.sin(Math.toRadians(45));
        return new double[][]{
                {-1040, -2250, -1050, -cs, 0, ss, 0, -1, 0},
                {1040, -2250, -1050, cs, 0, ss, 0, 1, 0},
                {0, -3570, -620, 0, -cf, sf, 1, 0, 0},
                {0, 760, -980, 0, cr, sr, -1, 0, 0},
        };
    }

    /** The 520-byte blob, in the layout of {@code /configure/camera_calibdata.bin}. */
    public static byte[] calibrationBytes() {
        ByteBuffer b = ByteBuffer.allocate(BirdseyeCalibration.BYTE_COUNT).order(ByteOrder.LITTLE_ENDIAN);
        b.putDouble(0); // header, uninterpreted
        for (double[] cam : rig()) {
            double[] c = {cam[0], cam[1], cam[2]}, f = {cam[3], cam[4], cam[5]}, r = {cam[6], cam[7], cam[8]};
            double[] d = cross(f, r);
            double[] rot = {r[0], r[1], r[2], d[0], d[1], d[2], f[0], f[1], f[2]}; // world -> camera rows
            double[] t = new double[3];
            for (int i = 0; i < 3; i++) t[i] = -(rot[i * 3] * c[0] + rot[i * 3 + 1] * c[1] + rot[i * 3 + 2] * c[2]);
            double[] rvec = rodriguesVector(rot);
            double[] fields = {FX, FY, CX, CY, 0, 0, K[0], K[1], K[2], K[3],
                    rvec[0], rvec[1], rvec[2], t[0], t[1], t[2]};
            for (double v : fields) b.putDouble(v);
        }
        return b.array();
    }

    private static double[] cross(double[] a, double[] b) {
        return new double[]{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    /** Rotation matrix (row-major) to axis * angle, robust near 180 degrees (via a quaternion). */
    static double[] rodriguesVector(double[] m) {
        double tr = m[0] + m[4] + m[8], w, x, y, z;
        if (tr > 0) {
            double s = Math.sqrt(tr + 1) * 2;
            w = s / 4; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
        } else if (m[0] > m[4] && m[0] > m[8]) {
            double s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
            w = (m[7] - m[5]) / s; x = s / 4; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
        } else if (m[4] > m[8]) {
            double s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
            w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = s / 4; z = (m[5] + m[7]) / s;
        } else {
            double s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
            w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = s / 4;
        }
        if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
        double n = Math.sqrt(x * x + y * y + z * z);
        if (n < 1e-15) return new double[]{0, 0, 0};
        double angle = 2 * Math.atan2(n, w);
        return new double[]{x / n * angle, y / n * angle, z / n * angle};
    }
}
