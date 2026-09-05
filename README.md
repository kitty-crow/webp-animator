# Animation Frame Aligner Web Tool

A tiny local browser front end around `anim_align_webp.py`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:18743
```

There is no Flask/FastAPI dependency. The web server uses Python's standard
library.

Upload multiple raster frames in animation order, choose the permitted
alignment direction and other settings, then click **Generate WebP**.

The generated `animation.webp` is returned directly to the browser and
downloaded automatically.

The backend does not resize or crop source pixels. If registration places a
frame outside the original frame bounds, the final animation canvas expands to
contain the complete translated frames.

## LAN access

The server binds to:

```text
0.0.0.0:18743
```

That means it listens on all network interfaces.

On the same computer:

```text
http://127.0.0.1:18743
```

From another device on the same LAN:

```text
http://<server-LAN-IP>:18743
```

For example:

```text
http://192.168.1.50:18743
```

Your operating-system firewall must allow inbound TCP connections on port 18743.
