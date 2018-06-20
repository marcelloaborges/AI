using System;

namespace Network
{
    public static class Maths
    {
        public static float[][] Standardizer(float[][] testSet)
        {
            var result = new float[testSet.Length][];

            var mean = Mean(testSet);

            var sd = StandardDeviation(testSet, mean);

            for (int i = 0; i < testSet.Length; i++)
            {
                result[i] = new float[testSet[i].Length];

                for (int j = 0; j < testSet[i].Length; j++)
                {
                    result[i][j] = (testSet[i][j] - mean) / sd;
                }
            }

            return result;
        }

        public static float Mean(float[][] data)
        {
            var sum = 0f;
            var mean = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    sum += data[i][j];
                }
            }

            mean = sum / (data.Length * data[0].Length);

            return mean;
        }

        public static float StandardDeviation(float[][] data, float mean = 0f)
        {
            if (Math.Abs(mean) < 0) Mean(data);

            var sum = 0f;
            var SD = 0f;
            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    sum += (float) Math.Pow(data[i][j] - mean, 2);
                }
            }

            SD = (float) Math.Sqrt(sum / (data.Length * data[0].Length));

            return SD;
        }
    }
}