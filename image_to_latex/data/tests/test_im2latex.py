import pytest

from image_to_latex.data.im2latex import Im2Latex


class TestIm2Latex:
    @pytest.fixture
    def im2latex(self):
        return Im2Latex()

    def test_setup_before_build_vocab(self, im2latex):
        with pytest.raises(Exception):
            im2latex.setup()
