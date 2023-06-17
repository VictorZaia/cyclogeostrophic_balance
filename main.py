
"""Packages"""

import PreProcessor as pre
import Processor as pro
import PostProcessor as post

"""Calling the functions that will compute the cyclogeostrophic balance"""

model = pre.PreProcessor.initialize_model()
pro.Processor.solve_model(model)
post.PostProcessor.write_results(model)
