using System;
using System.Collections.Generic;
using System.Linq;

namespace Network
{
    public class MLPSigmoidSoftmax
    {
        private readonly int _inputSize;
        private readonly int _hiddenSize;
        private readonly int _outputSize;
        private readonly float _weightsInitizalizationFactor;
        private readonly float _learningRate;

        private readonly float[,] _inputHidden;
        private readonly float[] _biasHidden;
        private readonly float[,] _hiddenOutput;
        private readonly float[] _biasOutput;

        public MLPSigmoidSoftmax(
            int inputSize,
            int hiddenSize,
            int outputSize,
            float weightsInitizalizationFactor = 1,
            float learningRate = .3f)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _outputSize = outputSize;
            _weightsInitizalizationFactor = weightsInitizalizationFactor;
            _learningRate = learningRate;

            _inputHidden = new float[inputSize, hiddenSize];
            _biasHidden = new float[hiddenSize];
            _hiddenOutput = new float[hiddenSize, outputSize];
            _biasOutput = new float[outputSize];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            var random = new Random();
            for (int j = 0; j < _hiddenSize; j++)
            {
                for (int i = 0; i < _inputSize; i++)
                {
                    _inputHidden[i, j] = (float) random.NextDouble() * _weightsInitizalizationFactor;
                }

                _biasHidden[j] = (float) random.NextDouble() * _weightsInitizalizationFactor;
            }

            for (int j = 0; j < _outputSize; j++)
            {
                for (int i = 0; i < _hiddenSize; i++)
                {
                    _hiddenOutput[i, j] = (float) random.NextDouble() * _weightsInitizalizationFactor;
                }

                _biasOutput[j] = (float) random.NextDouble() * _weightsInitizalizationFactor;
            }
        }

        public void Run(float[][] train, float[][] desired, string[][] test, int epoches = 1)
        {
            int epoch = 0;
            var r = new Random();

            while (epoch < epoches)
            {
                ////
                var tempList = new List<float[]>();
                var tempListDesired = new List<float[]>();

                var temp = train.ToList();
                var tempDesired = desired.ToList();

                while (temp.Count > 0)
                {
                    var i = r.Next(0, temp.Count);

                    tempList.Add(temp[i]);
                    temp.RemoveAt(i);

                    tempListDesired.Add(tempDesired[i]);
                    tempDesired.RemoveAt(i);
                }

                train = tempList.ToArray();
                desired = tempListDesired.ToArray();
                ////


                var error = 0f;
                for (int i = 0; i < train.Length; i++)
                {
                    var trainI = train[i];
                    var desiredI = desired[i];

                    var resultHidden = CalcHidden(trainI);

                    var resultOutput = CalcOutput(resultHidden);

                    error = Learn(trainI, resultHidden, resultOutput, desiredI);
                }

                if (epoch % 1000 == 0)
                {
                    Console.Clear();

                    Console.WriteLine($"Error: {error}");

                    for (int i = 0; i < test.Length; i++)
                    {
                        var testI = test[i];
                        var input = new[]
                        {
                            float.Parse(testI[0]),
                            float.Parse(testI[1]),
                            float.Parse(testI[2])
                        };

                        CalcAndPrintResults(input, i, testI[testI.Length - 1]);
                    }
                }

                epoch++;
            }
        }

        private void CalcAndPrintResults(float[] input, int testNumber, string msg)
        {
            var resultHidden = CalcHidden(input);

            var resultOutput = CalcOutput(resultHidden);

            Console.WriteLine($"Test {testNumber}: {msg}");
            Console.WriteLine($"WHITE    {resultOutput[0]}");
            Console.WriteLine($"BLACK    {resultOutput[1]}");
            Console.WriteLine($"RED      {resultOutput[2]}");
            Console.WriteLine($"GREEN    {resultOutput[3]}");
            Console.WriteLine($"BLUE     {resultOutput[4]}");
            Console.WriteLine($"YELLOW   {resultOutput[5]}");

            Console.WriteLine();
        }

        private float[] CalcHidden(float[] input)
        {
            var result = new float[_hiddenSize];

            for (int j = 0; j < _hiddenSize; j++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    result[j] += input[i] * _inputHidden[i, j];
                }

                //result[j] += _biasHidden[j];
            }

            for (int i = 0; i < _hiddenSize; i++)
            {
                result[i] = Sigmoid(result[i]);
            }

            return result;
        }

        private float[] CalcOutput(float[] input)
        {
            var result = new float[_outputSize];

            for (int j = 0; j < _outputSize; j++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    result[j] += input[i] * _hiddenOutput[i, j];
                }

                //result[j] += _biasOutput[j];
            }

            result = Softmax(result);
            //for (int i = 0; i < _outputSize; i++)
            //{
            //    result[i] = Sigmoid(result[i]);
            //}

            return result;
        }

        private float Sigmoid(float x)
        {
            var parcial = (float) Math.Exp(-1 * x);

            return 1 / (1 + parcial);
        }

        private float DerivativeSigmoid(float x)
        {
            var sigmoid = Sigmoid(x);

            return sigmoid * (1 - sigmoid);
        }

        private float[] Softmax(float[] x)
        {
            var result = new float[x.Length];

            var expX = new float[x.Length];

            var sumExpX = 0f;
            for (int i = 0; i < x.Length; i++)
            {
                expX[i] = (float) Math.Exp(x[i]);
                sumExpX += expX[i];
            }

            for (int i = 0; i < x.Length; i++)
            {
                result[i] = expX[i] / sumExpX;
            }

            return result;
        }

        private float Learn(float[] input, float[] resultHidden, float[] resultOutput, float[] desired)
        {
            //ERROR OUTPUT

            //SIGMOID 

            //for (int i = 0; i < _outputSize; i++)
            //{
            //    errorOutput[i] = desired[i] - resultOutput[i];
            //}

            //SOFTMAX
            var errorOutput = resultOutput;

            var indexDesired = 0;

            for (int i = 0; i < desired.Length; i++)
            {
                if (Math.Abs(desired[i] - 1) < 0.01)
                {
                    indexDesired = i;
                    break;
                }
            }

            errorOutput[indexDesired] -= 1;

            //DERIVATIVE OUTPUT

            var dOutput = new float[_outputSize];

            //SIGMOID
            //for (int i = 0; i < _outputSize; i++)
            //{                
            //    dOutput[i] = errorOutput[i] * DerivativeSigmoid(resultOutput[i]);
            //}

            //SOFTMAX
            for (int i = 0; i < _outputSize; i++)
            {
                dOutput[i] = errorOutput[i] * 1; //SOFTMAX CALCs THE ERROR AND DERIVATIVE IN ONE STEP
            }

            //ERROR HIDDEN

            var errorHidden = new float[_hiddenSize];

            for (int i = 0; i < _hiddenSize; i++)
            {
                for (int j = 0; j < _outputSize; j++)
                {
                    errorHidden[i] += _hiddenOutput[i, j] * dOutput[j];
                }
            }

            //DERIVATIVE HIDDEN

            var dHidden = new float[_hiddenSize];

            for (int i = 0; i < _hiddenSize; i++)
            {
                dHidden[i] = errorHidden[i] * DerivativeSigmoid(resultHidden[i]);
            }

            //GRADIENT => derivative * input

            var gOutput = new float[_hiddenSize, _outputSize];
            for (int i = 0; i < _outputSize; i++)
            {
                for (int j = 0; j < _hiddenSize; j++)
                {
                    gOutput[j, i] = dOutput[i] * resultHidden[j];
                }
            }

            var gHidden = new float[_inputSize, _hiddenSize];
            for (int i = 0; i < _hiddenSize; i++)
            {
                for (int j = 0; j < _inputSize; j++)
                {
                    gHidden[j, i] = dHidden[i] * input[j];
                }
            }

            //UPDATE WEIGHTs

            for (int i = 0; i < _inputSize; i++)
            {
                for (int j = 0; j < _hiddenSize; j++)
                {
                    _inputHidden[i, j] -= gHidden[i, j] * _learningRate;
                }
            }

            for (int i = 0; i < _hiddenSize; i++)
            {
                for (int j = 0; j < _outputSize; j++)
                {
                    _hiddenOutput[i, j] -= gOutput[i, j] * _learningRate;
                }
            }

            //LOSS

            var error = 0f;

            for (int i = 0; i < _outputSize; i++)
            {
                error += desired[i] - resultOutput[i];
            }

            error = error / _outputSize;

            return error;
            //return EuclidianDistance(input, resultOutput);
        }

        private float EuclidianDistance(float[] a, float[] b)
        {
            var result = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                result += (float) Math.Pow(a[i] - b[i], 2);
            }

            return (float) Math.Sqrt(result);
        }
    }
}