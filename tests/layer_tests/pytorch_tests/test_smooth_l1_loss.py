# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestSmoothL1Loss(PytorchLayerTest):
    def _prepare_input(self):
        # Generate random input data with smaller values to avoid FP16 overflow
        input_shape = (1, 3, 224, 224)
        return (np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32),
                np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32))

    def create_model(self, reduction="mean", beta=1.0):
        class aten_smooth_l1_loss(torch.nn.Module):
            def __init__(self, reduction, beta):
                super(aten_smooth_l1_loss, self).__init__()
                self.reduction = reduction
                self.beta = beta

            def forward(self, input, target):
                return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

        ref_net = None
        # PyTorch may lower smooth_l1_loss to aten::l1_loss when beta==0, accept both
        kind = "aten::l1_loss" if beta == 0.0 else "aten::smooth_l1_loss"
        return aten_smooth_l1_loss(reduction, beta), ref_net, kind

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
    def test_smooth_l1_loss(self, reduction, beta, ie_device, precision, ir_version):
        self._test(*self.create_model(reduction, beta), ie_device, precision, ir_version)

    @pytest.mark.precommit
    def test_smooth_l1_equal_huber_beta1(self, ie_device, precision, ir_version):
        """
        בדיקה פשוטה: מאמתת ש-SmoothL1Loss ו-HuberLoss זהים כאשר beta == 1.0
        """
        class M(torch.nn.Module):
            def forward(self, x, y):
                # SmoothL1 של פייתורץ (beta=1) עם reduction=mean (ברירת המחדל)
                s = F.smooth_l1_loss(x, y, beta=1.0)

                # Huber ידני (delta == beta == 1.0), reduction=mean
                beta = 1.0
                diff = x - y
                abs_diff = torch.abs(diff)

                # piecewise:
                # |d| < beta  ->  0.5 * d^2 / beta
                # אחרת        ->  |d| - 0.5 * beta
                per_elem = torch.where(
                    abs_diff < beta,
                    0.5 * (diff ** 2) / beta,
                    abs_diff - 0.5 * beta
                )

                h = per_elem.mean()  # התאמה לברירת המחדל של F.huber_loss

                # אמורים להיות שווים כאשר beta==1
                return s - h  # ההפרש אמור להיות אפס

        model = M().eval()
        x = np.random.randn(8, 8).astype(np.float32)
        y = np.random.randn(8, 8).astype(np.float32)

        # מצפים שההפרש יהיה קרוב מאוד לאפס בכל דיוק/מכשיר
        self._test(model, None, "aten::sub", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_beta_zero_is_l1(self, reduction, ie_device, precision, ir_version):
        """
        כש-beta==0, SmoothL1 אמור להיות בדיוק L1 (|x - y|) עם אותו reduction.
        כדי להימנע מתלות ב-aten::l1_loss, מחשבים L1 ידנית בגרף.
        """
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction

            def forward(self, x, y):
                s = F.smooth_l1_loss(x, y, beta=0.0, reduction=self.reduction)
                l1_elems = torch.abs(x - y)
                if self.reduction == "none":
                    l1 = l1_elems
                elif self.reduction == "mean":
                    l1 = l1_elems.mean()
                else:  # "sum"
                    l1 = l1_elems.sum()
                return s - l1  # אמור להיות 0

        model = M(reduction).eval()
        x = np.random.randn(8, 16).astype(np.float32)
        y = np.random.randn(8, 16).astype(np.float32)

        # המודל מחזיר s - l1; מצפים לאפס (או קרוב מאוד) בשני הצדדים (PyTorch ו-OV).
        self._test(model, None, "aten::sub", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_nan_propagation(self, reduction, ie_device, precision, ir_version):
        """
        מאמת הפצת NaN: אם יש NaN ב-input/target, הפלט צריך להיות NaN באותם מקומות (none),
        או NaN כסקלר (mean/sum), בדומה לפייתורץ.
        """
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction
            def forward(self, x, y):
                # beta=1.0 (לא משנה מבחינת NaN), נשאר עם ברירת-המחדל reduction שנמסר
                return F.smooth_l1_loss(x, y, reduction=self.reduction, beta=1.0)

        model = M(reduction).eval()

        # קלט קטן עם NaN באמצע כדי לזהות מיקום ב-"none" ולבדוק סכאלר ב-"mean/sum"
        x = np.array([0.0, np.nan, 1.0], dtype=np.float32)
        y = np.array([0.0, 0.0,   2.0], dtype=np.float32)

        def _compare(ov_outs, pt_outs):
            # ההארנס מחזיר רשימות/מערכים; נהפוך ל-np.array לשני הצדדים
            ov = np.asarray(ov_outs)
            pt = np.asarray(pt_outs)

            if reduction == "none":
                # ודא מיקום ה-NaN זהה
                np.testing.assert_array_equal(np.isnan(ov), np.isnan(pt))
                # על הערכים שאינם NaN – השוואה נומרית רגילה
                mask = ~np.isnan(pt)
                np.testing.assert_allclose(ov[mask], pt[mask], rtol=1e-5, atol=1e-5)
            else:
                # mean/sum אמורים להיות סקלריים: שניהם NaN
                assert np.isnan(ov).item() and np.isnan(pt).item()

        # מצפים לאופרטור smooth_l1 עצמו; השוואה דרך compare_func כדי לא "ליפול" על NaN ב-allclose
        self._test(model, None, "aten::smooth_l1_loss",
                   ie_device, precision, ir_version, inputs=[x, y], compare_func=_compare)

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("int_dtype", [np.int32, np.int64])
    def test_smooth_l1_dtype_alignment_int_target(self, reduction, int_dtype, ie_device, precision, ir_version):
        """
        מוודא שכאשר target הוא שלם (int32/64) והקלט float32, המרה ל-dtype של input מתבצעת בדיוק כמו בפייתורץ.
        הבדיקה מחזירה s - s_ref ומצפה ל-0:
          s     = smooth_l1_loss(x, y_int)
          s_ref = smooth_l1_loss(x, y_int.astype(x.dtype))
        """
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction

            def forward(self, x, y_int):
                s = F.smooth_l1_loss(x, y_int, reduction=self.reduction, beta=1.0)
                s_ref = F.smooth_l1_loss(x, y_int.to(x.dtype), reduction=self.reduction, beta=1.0)
                return s - s_ref  # אמור להיות 0 אם ה-convert מתבצע כמו שצריך

        model = M(reduction).eval()

        # input כ-float32 (בטוח לריצה על CPU; ב-OV ה-precision נשלט ע"י הפרמטרים החיצוניים)
        x = np.random.randn(4, 5).astype(np.float32)
        # target שלם (int32/64) – מדמה מקרה נפוץ בו ground truth בא באינדקסים/מספרים שלמים
        y = (np.random.randn(4, 5) * 10).astype(int_dtype)

        # מצפים ל-aten::sub כי המודל מחזיר הפרש
        self._test(model, None, "aten::sub", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.nightly
    @pytest.mark.parametrize("beta", [1e-6, 1e6])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_beta_extremes(self, beta, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, beta, reduction):
                super().__init__()
                self.beta = beta
                self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, beta=self.beta, reduction=self.reduction)

        model = M(beta, reduction).eval()
        x = np.random.uniform(-0.05, 0.05, size=(16, 16)).astype(np.float32)
        y = np.random.uniform(-0.05, 0.05, size=(16, 16)).astype(np.float32)
        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_inf_propagation(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, reduction=self.reduction, beta=1.0)

        model = M(reduction).eval()
        x = np.array([0.0, np.inf, 1.0, -np.inf], dtype=np.float32)
        y = np.array([0.0, 0.0,  2.0,  3.0   ], dtype=np.float32)

        def _cmp(ov_outs, pt_outs):
            ov = np.asarray(ov_outs)
            pt = np.asarray(pt_outs)
            if reduction == "none":
                np.testing.assert_array_equal(np.isinf(ov), np.isinf(pt))
            else:
                assert np.isinf(ov).item() and np.isinf(pt).item()

        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y], compare_func=_cmp)

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_output_dtype_matches_input(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, beta=1.0, reduction=self.reduction)

        model = M(reduction).eval()
        x = np.random.randn(4, 5).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)

        def _cmp(ov_outs, pt_outs):
            ov = np.asarray(ov_outs)
            pt = np.asarray(pt_outs)
            # ensure numeric equality and matching shape/structure
            np.testing.assert_allclose(ov, pt, rtol=1e-5, atol=1e-5)

        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y], compare_func=_cmp)

    @pytest.mark.precommit
    def test_smooth_l1_negative_beta_raises(self, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, beta=-0.5)

        model = M().eval()
        x = np.random.randn(2, 3).astype(np.float32)
        y = np.random.randn(2, 3).astype(np.float32)

        with pytest.raises(Exception):
            self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_broadcasting(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__(); self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, reduction=self.reduction, beta=0.5)
        model = M(reduction).eval()
        x = np.random.uniform(-0.1, 0.1, size=(1,3,224,224)).astype(np.float32)
        y = np.random.uniform(-0.1, 0.1, size=(1,3,1,1)).astype(np.float32)  # broadcast to x
        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("beta_val", [1.0, 1e-8])  # regular and very small edge
    @pytest.mark.xfail(reason="PyTorch F.smooth_l1_loss expects float beta; passing tensor beta is not representable in TorchScript graph. Dynamic beta as tensor is currently unsupported in TS path.")
    def test_smooth_l1_dynamic_beta_tensor(self, beta_val, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def forward(self, x, y, beta):
                # Note: F.smooth_l1_loss requires a float beta; TS cannot represent a tensor-to-float cast here.
                return F.smooth_l1_loss(x, y, beta=1.0)
        model = M().eval()
        x = np.random.randn(8, 8).astype(np.float32)
        y = np.random.randn(8, 8).astype(np.float32)
        beta = np.array(beta_val, dtype=np.float32)  # 0-d tensor (unused due to TS limitation)
        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y, beta])

    @pytest.mark.precommit
    @pytest.mark.parametrize("red_code, red_str", [(0, "none"), (1, "mean"), (2, "sum")])
    def test_smooth_l1_reduction_from_input_code(self, red_code, red_str, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, red_str):
                super().__init__(); self.red = red_str
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, beta=0.5, reduction=self.red)
        model = M(red_str).eval()
        x = np.random.randn(4, 5).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y])

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_l1_loss_basic(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction): super().__init__(); self.reduction = reduction
            def forward(self, x, y): return F.l1_loss(x, y, reduction=self.reduction)
        model = M(reduction).eval()
        x = np.random.randn(6, 7).astype(np.float32)
        y = np.random.randn(6, 7).astype(np.float32)
        self._test(model, None, "aten::l1_loss", ie_device, precision, ir_version, inputs=[x, y])
        