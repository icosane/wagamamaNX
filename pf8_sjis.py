#!/usr/bin/env python3
"""
pf8_sjis.py - Pack/unpack Artemis .pf8/.pfs archives with Shift-JIS filename support.

This is an adaptation of the well-known artemis_pf8.py reference tool
(YuriSizuku/GalgameReverse, project/artemis/src/artemis_pf8.py), modified so
that filenames inside the archive index are encoded/decoded using Shift-JIS
(cp932) instead of UTF-8. This matches what the original Artemis engine
(running on a Japanese Windows codepage) expects, and is what pfs-rs
currently does NOT do when *creating* archives.

PF8 format (for reference):
    magic 'pf8'                (3 bytes)
    index_size          u32 LE   (counted starting right after this field)
    index_count         u32 LE
    file_entries[index_count]:
        name_length     u32 LE
        name            name_length bytes (Shift-JIS encoded, NUL-padded)
        reserved        4 bytes (00 00 00 00)
        offset          u32 LE  (absolute offset of file data)
        size            u32 LE
    filesize_count      u32 LE   (= index_count + 1)
    filesize_offsets[filesize_count]  u64 LE  (last entry is 0)
    filesize_count_offset u32 LE
    -- file data follows, XOR-encrypted with SHA1(index_bytes) as the key,
       except for files matching `unencrypted_filter` --

Usage:
    Pack a directory into an archive with Shift-JIS filenames:
        python3 pf8_sjis.py pack <input_dir> <output.pfs>

    Unpack an archive, decoding filenames with Shift-JIS (falls back to
    UTF-8 automatically if a name happens to be valid UTF-8):
        python3 pf8_sjis.py unpack <input.pfs> <output_dir>

    List contents without extracting:
        python3 pf8_sjis.py list <input.pfs>

Notes:
- Internal path separators are written as backslashes ('\\'), matching what
  the original Windows-built Artemis engine expects. Use --unix-sep if you
  specifically need forward slashes.
- Files whose name cannot be represented in Shift-JIS will raise a clear
  error before anything is written, rather than silently corrupting names.
"""

import os
import re
import sys
import struct
import hashlib
import argparse
from io import BytesIO

DEFAULT_UNENCRYPTED = [r'\.mp4$', r'\.flv$', r'\.ogg$']


# ---------------------------------------------------------------------------
# Core encryption helpers (unchanged from the reference implementation)
# ---------------------------------------------------------------------------

def make_key(index_data: bytes) -> bytes:
    return hashlib.sha1(index_data).digest()


def xor_crypt(buf: bytearray, start: int, size: int, key: bytes) -> None:
    """In-place XOR of buf[start:start+size] with the repeating key."""
    klen = len(key)
    for i in range(size):
        buf[start + i] ^= key[i % klen]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def encode_name(name: str, encoding: str) -> bytes:
    try:
        return name.encode(encoding)
    except UnicodeEncodeError as e:
        raise ValueError(
            f"Filename {name!r} contains characters that cannot be encoded "
            f"as {encoding}: {e}"
        ) from e


def collect_files(input_dir: str, unix_sep: bool):
    filelist = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, input_dir)
            if unix_sep:
                rel = rel.replace(os.sep, '/')
            else:
                rel = rel.replace('/', '\\').replace(os.sep, '\\')
            size = os.path.getsize(full)
            filelist.append((rel, size, full))
    return filelist


def pack(input_dir: str, output_path: str, encoding: str = 'shift_jis',
          unencrypted_filter=None, unix_sep: bool = False,
          verbose: bool = True) -> None:
    if unencrypted_filter is None:
        unencrypted_filter = DEFAULT_UNENCRYPTED

    filelist = collect_files(input_dir, unix_sep)
    if not filelist:
        raise ValueError(f"No files found under {input_dir!r}")

    # Pre-encode names once so we fail fast on any un-encodable filename.
    encoded_names = [encode_name(name, encoding) for name, _, _ in filelist]

    fileentry_size = sum(len(nb) + 16 for nb in encoded_names)
    index_count = len(filelist)
    index_size = 0x4 + fileentry_size + 0x4 + (index_count + 1) * 0x8 + 0x4

    buf = BytesIO()
    buf.write(b'pf8')
    buf.write(struct.pack('<II', index_size, index_count))

    file_offset = index_size + 0x7
    filesize_offsets = []
    entries = []  # (rel_name, offset, size) for logging

    for (rel, size, _full), name_bytes in zip(filelist, encoded_names):
        buf.write(struct.pack('<I', len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack('<III', 0x0, file_offset, size))
        filesize_offsets.append(buf.tell() - 0x4 - 0xF)
        entries.append((rel, file_offset, size))
        file_offset += size

    buf.write(struct.pack('<I', index_count + 1))
    filesize_count_offset = buf.tell() - 0x4 - 0x7
    for off in filesize_offsets:
        buf.write(struct.pack('<Q', off))
    buf.write(struct.pack('<QI', 0x0, filesize_count_offset))

    if buf.tell() - 0x7 != index_size:
        raise AssertionError(
            f"internal error: index size mismatch "
            f"({buf.tell() - 0x7} != {index_size})"
        )

    # Append file contents
    for rel, _off, _size, full in [(r, o, s, f) for (r, s, f), o in
                                     zip(filelist, (e[1] for e in entries))]:
        with open(full, 'rb') as fp:
            data = fp.read()
        if len(data) != _size:
            raise AssertionError(f"size changed while reading {full!r}")
        buf.write(data)
        if verbose:
            print(f"added: {rel} ({_size} bytes)")

    # Encrypt
    data = bytearray(buf.getvalue())
    index_data = bytes(data[0x7:0x7 + index_size])
    key = make_key(index_data)
    if verbose:
        print(f"key (sha1 of index) = {key.hex()}")

    re_unencrypted = [re.compile(p, re.IGNORECASE) for p in unencrypted_filter]

    for rel, offset, size in entries:
        skip = any(p.search(rel) for p in re_unencrypted)
        if not skip:
            xor_crypt(data, offset, size, key)
        if verbose:
            state = "stored" if skip else "encrypted"
            print(f"{state}: {rel} @0x{offset:X} size={size}")

    with open(output_path, 'wb') as fp:
        fp.write(data)

    if verbose:
        print(f"\nWrote {output_path} ({len(data)} bytes), "
              f"{index_count} files, names encoded as {encoding}.")


# ---------------------------------------------------------------------------
# Unpacking / listing
# ---------------------------------------------------------------------------

def decode_name(name_bytes: bytes, encoding: str) -> str:
    name_bytes = name_bytes.split(b'\x00', 1)[0]  # strip NUL padding
    # Try requested encoding first, then fall back to UTF-8 / lossy so we
    # never crash on an unexpected archive.
    for enc in (encoding, 'utf-8'):
        try:
            return name_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return name_bytes.decode('shift_jis', errors='replace')


def parse_index(data: bytes, encoding: str):
    if data[0:3] != b'pf8':
        raise ValueError("not a pf8 archive (bad magic)")
    index_size, index_count = struct.unpack('<II', data[3:11])
    entries = []
    cur = 11
    for _ in range(index_count):
        name_length = struct.unpack('<I', data[cur:cur + 4])[0]
        name_bytes = data[cur + 4:cur + 4 + name_length]
        name = decode_name(name_bytes, encoding)
        cur += 4 + name_length
        cur += 4  # reserved
        offset, size = struct.unpack('<II', data[cur:cur + 8])
        cur += 8
        entries.append({'name': name, 'offset': offset, 'size': size})
    index_data = data[0x7:0x7 + index_size]
    return entries, index_data


def unpack(input_path: str, output_dir: str, encoding: str = 'shift_jis',
           unencrypted_filter=None, verbose: bool = True) -> None:
    if unencrypted_filter is None:
        unencrypted_filter = DEFAULT_UNENCRYPTED

    with open(input_path, 'rb') as fp:
        data = fp.read()

    entries, index_data = parse_index(data, encoding)
    key = make_key(index_data)
    re_unencrypted = [re.compile(p, re.IGNORECASE) for p in unencrypted_filter]

    for e in entries:
        name, offset, size = e['name'], e['offset'], e['size']
        skip = any(p.search(name) for p in re_unencrypted)
        chunk = bytearray(data[offset:offset + size])
        if not skip:
            xor_crypt(chunk, 0, size, key)
        rel = name.replace('\\', os.sep).replace('/', os.sep)
        full = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(full) or '.', exist_ok=True)
        with open(full, 'wb') as fp2:
            fp2.write(chunk)
        if verbose:
            print(f"extracted: {name} ({size} bytes)")


def list_archive(input_path: str, encoding: str = 'shift_jis') -> None:
    with open(input_path, 'rb') as fp:
        data = fp.read()
    entries, _ = parse_index(data, encoding)
    total = 0
    for e in entries:
        print(f"{e['name']}\t{e['size']} bytes\t@0x{e['offset']:X}")
        total += e['size']
    print(f"\n{len(entries)} files, {total} bytes total")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    pp = sub.add_parser('pack', help='Pack a directory into a .pfs archive')
    pp.add_argument('input_dir')
    pp.add_argument('output_pfs')
    pp.add_argument('--encoding', default='shift_jis',
                     help="filename encoding to use (default: shift_jis)")
    pp.add_argument('--unix-sep', action='store_true',
                     help="use '/' instead of '\\\\' as the path separator")
    pp.add_argument('--quiet', action='store_true')

    up = sub.add_parser('unpack', help='Unpack a .pfs archive')
    up.add_argument('input_pfs')
    up.add_argument('output_dir')
    up.add_argument('--encoding', default='shift_jis',
                     help="filename encoding to try first (default: shift_jis)")
    up.add_argument('--quiet', action='store_true')

    lp = sub.add_parser('list', help='List archive contents')
    lp.add_argument('input_pfs')
    lp.add_argument('--encoding', default='shift_jis')

    args = p.parse_args()

    if args.command == 'pack':
        pack(args.input_dir, args.output_pfs, encoding=args.encoding,
             unix_sep=args.unix_sep, verbose=not args.quiet)
    elif args.command == 'unpack':
        unpack(args.input_pfs, args.output_dir, encoding=args.encoding,
               verbose=not args.quiet)
    elif args.command == 'list':
        list_archive(args.input_pfs, encoding=args.encoding)


if __name__ == '__main__':
    main()
