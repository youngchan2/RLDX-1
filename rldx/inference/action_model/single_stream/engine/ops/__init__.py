"""Register all custom ops for SingleStreamBlock.

Import this module to register all ss:: ops at once.
"""

from __future__ import annotations

import single_stream.engine.ops.op_fused_attention  # ss::fused_attention_2way
import single_stream.engine.ops.op_fused_attention_3way  # ss::fused_attention_3way
import single_stream.engine.ops.op_fused_epilogue_ln  # ss::fused_epilogue_ln
import single_stream.engine.ops.op_libra_attention  # ss::libra_fused_attention_2way
import single_stream.engine.ops.op_libra_attention_3way # ss::libra_fused_attention_3way
