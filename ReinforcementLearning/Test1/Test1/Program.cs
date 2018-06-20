using System;

namespace Test1
{
    class Program
    {
        static double bandit1Real = 0.8d;
        static double bandit2Real = 0.4d;
        static double bandit3Real = 0.3d;

        static void Main(string[] args)
        {
            //bandit1 Real => 0.80d;
            //bandit2 Real => 0.40d;
            //bandit3 Real => 0.30d;

            var bandit1Id = 1;
            var bandit1 = 0.4d;
            var bandit2Id = 2;
            var bandit2 = 0.3d;
            var bandit3Id = 3;
            var bandit3 = 0.3d;
            var keep = 0.5d;

            Random r = new Random();

            var pulls = 0;
            var wins = 0;

            while (pulls < 1000)
            {
                var keepChance = r.NextDouble();

                Result result;

                if (keepChance <= keep) result = Keep(bandit1Id, bandit1, bandit2Id, bandit2, bandit3Id, bandit3);
                else result = Explore(bandit1Id, bandit1, bandit2Id, bandit2, bandit3Id, bandit3);

                var update = new double[4];
                if (result.Win) wins++;

                update = Update(result, keepChance <= keep, keep, bandit1Id, bandit1, bandit2Id, bandit2, bandit3Id, bandit3);

                bandit1 += update[0];
                bandit2 += update[1];
                bandit3 += update[2];
                keep += update[3];

                pulls++;

                if (pulls % 100 == 0)
                {
                    Console.WriteLine($"{pulls} => {wins}");
                    Console.WriteLine($"A: {bandit1}");
                    Console.WriteLine($"B: {bandit2}");
                    Console.WriteLine($"C: {bandit3}");
                    Console.WriteLine($"Keep: {keep}");
                }
            }
        }

        static double[] gradient = new double[4];

        private static double[] Update(
            Result result,
            bool keep,
            double keepChance,
            int bandit1Id,
            double bandit1Chance,
            int bandit2Id,
            double bandit2Chance,
            int bandit3Id,
            double bandit3Chance)
        {
            var learningRate = 0.01d;
            var winBonus = 0.01d;

            var newValues = new double[4];

            if (result.Win)
            {
                if (result.Id == bandit1Id)
                {
                    newValues[0] = (1 - bandit1Chance) * learningRate;
                    newValues[1] = (-bandit2Chance) * learningRate;
                    newValues[2] = (-bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance - 1, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance - 1);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance);
                }
                if (result.Id == bandit2Id)
                {
                    newValues[0] = (-bandit1Chance) * learningRate;
                    newValues[1] = (1 - bandit2Chance) * learningRate;
                    newValues[2] = (-bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance - 1, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance - 1);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance);
                }
                if (result.Id == bandit3Id)
                {
                    newValues[0] = (-bandit1Chance) * learningRate;
                    newValues[1] = (-bandit2Chance) * learningRate;
                    newValues[2] = (1 - bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance - 1, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance - 1);
                }
            }
            else
            {
                if (result.Id == bandit1Id)
                {
                    newValues[0] = (-bandit1Chance) * learningRate;
                    newValues[1] = (1 - bandit2Chance) * learningRate;
                    newValues[2] = (1 - bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance - 1, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance - 1);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance - 1, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance - 1);
                }
                if (result.Id == bandit2Id)
                {
                    newValues[0] = (1 - bandit1Chance) * learningRate;
                    newValues[1] = (-bandit2Chance) * learningRate;
                    newValues[2] = (1 - bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance - 1, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance - 1);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance - 1, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance - 1);
                }
                if (result.Id == bandit3Id)
                {
                    newValues[0] = (1 - bandit1Chance) * learningRate;
                    newValues[1] = (1 - bandit2Chance) * learningRate;
                    newValues[2] = (-bandit3Chance) * learningRate;
                    //gradient[0] = 0.999 * gradient[0] + 0.001 * Math.Pow(bandit1Chance - 1, 2);
                    //newValues[0] = (learningRate / (Math.Sqrt(gradient[0]))) * (bandit1Chance - 1);

                    //gradient[1] = 0.999 * gradient[1] + 0.001 * Math.Pow(bandit2Chance - 1, 2);
                    //newValues[1] = (learningRate / (Math.Sqrt(gradient[1]))) * (bandit2Chance - 1);

                    //gradient[2] = 0.999 * gradient[2] + 0.001 * Math.Pow(bandit3Chance, 2);
                    //newValues[2] = (learningRate / (Math.Sqrt(gradient[2]))) * (bandit3Chance);
                }
            }

            if (keep)
            {
                if (result.Win)
                {
                    newValues[3] = (1 - keepChance) * learningRate;
                    //gradient[3] = 0.999 * gradient[3] + 0.001 * Math.Pow(keepChance - 1, 2);
                    //newValues[3] = (learningRate / (Math.Sqrt(gradient[3]))) * (keepChance - 1);
                }
                else
                {
                    newValues[3] = -keepChance * learningRate;
                    //gradient[3] = 0.999 * gradient[3] + 0.001 * Math.Pow(keepChance, 2);
                    //newValues[3] = (learningRate / (Math.Sqrt(gradient[3]))) * (keepChance);
                }
            }

            return newValues;
        }

        private class Result
        {
            public int Id { get; set; }
            public bool Win { get; set; }
        }

        private static bool Play(int banditId)
        {
            Random r = new Random();

            var chance = r.NextDouble();

            var banditChange = 0d;

            if (banditId == 1) banditChange = bandit1Real;
            if (banditId == 2) banditChange = bandit2Real;
            if (banditId == 3) banditChange = bandit3Real;

            if (chance < banditChange) return true;

            return false;
        }

        private static Result Keep(int bandit1Id, double bandit1Chance, int bandit2Id, double bandit2Chance, int bandit3Id, double bandit3Chance)
        {
            var biggestId = bandit1Id;
            var biggestChange = bandit1Chance;

            if (bandit2Chance > biggestChange)
            {
                biggestId = bandit2Id;
                biggestChange = bandit2Chance;
            }
            if (bandit3Chance > biggestChange)
            {
                biggestId = bandit3Id;
                biggestChange = bandit3Chance;
            }

            return new Result()
            {
                Id = biggestId,
                Win = Play(biggestId)
            };
        }

        private static Result Explore(int bandit1Id, double bandit1Chance, int bandit2Id, double bandit2Chance, int bandit3Id, double bandit3Chance)
        {
            var biggestId = 1;
            var biggestChange = bandit1Chance;

            if (bandit2Chance > biggestChange)
            {
                biggestId = 2;
                biggestChange = bandit2Chance;
            }
            if (bandit3Chance > biggestChange)
            {
                biggestId = 3;
                biggestChange = bandit3Chance;
            }

            var chosenBandit1Id = 0;
            var chosenbandit2Id = 0;

            if (biggestId == 1)
            {
                chosenBandit1Id = bandit2Id;
                chosenbandit2Id = bandit3Id;
            }

            if (biggestId == 2)
            {
                chosenBandit1Id = bandit1Id;
                chosenbandit2Id = bandit3Id;
            }

            if (biggestId == 3)
            {
                chosenBandit1Id = bandit1Id;
                chosenbandit2Id = bandit2Id;
            }

            Random r = new Random();

            var choice = r.NextDouble();

            if (choice >= 0.5d)
            {
                return new Result()
                {
                    Id = chosenBandit1Id,
                    Win = Play(chosenBandit1Id)
                };
            }
            else
            {
                return new Result()
                {
                    Id = chosenbandit2Id,
                    Win = Play(chosenbandit2Id)
                };
            }
        }

    }
}
