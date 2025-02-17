import setuptools


__packagename__ = 'transformers-openai'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8',
    description='OpenAI compatibility using FastAPI HuggingFace Transformers.',
    author='huseinzol05',
    url='https://github.com/malaysia-ai/transformers-openai-api',
    include_package_data=True
)
