# Third-party code and data

TS2Vec's encoder, dilated convolutions and contrastive losses are included under
the [MIT license](src/trd/hosts/ts2vec/vendor/LICENSE). The source revision is
recorded in [UPSTREAM.md](src/trd/hosts/ts2vec/vendor/UPSTREAM.md). The adapter
adds the reversal view and exposes the averaged encoder's final timestamp
features. Upstream: <https://github.com/yuezhihan/ts2vec>.

PatchTST's numerical backbone and patch masking code are included under the
[Apache 2.0 license](src/trd/hosts/patchtst/vendor/LICENSE), from
<https://github.com/yuqinie98/PatchTST>, revision
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`. Local changes provide explicit padding
masks, dense feature extraction and separate normalization for native/TRD views.

Dataset files are obtained separately from their publishers. CirCor's release
and terms are available on [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/).
OpenPack 1.0.0 is available from [Zenodo](https://zenodo.org/records/8145223),
with its [dataset documentation](https://open-pack.github.io/) and
[CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Public recording and participant identifiers are retained in split manifests
to make the reported partitioning reproducible.

The first-party code license is pending author selection before public release.
This does not alter the licenses of the vendored code or datasets.
