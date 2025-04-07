import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

print(tf.config.list_physical_devices('GPU'))
print(tf.test.gpu_device_name())
print(tf.add(tf.ones([2,2]), tf.ones([2,2])).numpy())
print(tf.test.is_built_with_rocm())
class MyTest(tf.test.TestCase):

  def test_add_on_gpu(self):
    if not tf.test.is_built_with_rocm():
      self.skipTest("test is only applicable on GPU")

    with tf.device("GPU:0"):
      self.assertEqual(tf.math.add(1.0, 2.0), 3.0)
t = MyTest()
t.test_add_on_gpu()