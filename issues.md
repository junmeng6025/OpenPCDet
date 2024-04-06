
### ImportError: cannot import name 'builder' from 'google.protobuf.internal'
https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal

Follow these steps:

- Install the latest protobuf version (in my case is 4.21.1)
    ```bash
    pip install --upgrade protobuf
    ```

- Copy `builder.py` from `.../Lib/site-packages/google/protobuf/internal` to another folder on your computer (let's say `/Documents`)
    > In Docker: `/usr/local/lib/python3.8/dist-packages/google/protobuf/internal`
- Install a protobuf version that is compatible with your project (for me 3.19.4)
```bash
pip install protobuf==3.19.4
```
- Copy `builder.py` from (let's say `Documents`) to `/usr/local/lib/puthon3.8/dist-packages/google/protobuf/internal` to override the older version
- Run your code