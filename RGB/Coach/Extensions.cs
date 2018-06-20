using System;
using System.Collections.Generic;
using System.Security.Cryptography;

namespace Coach
{
    public static class Extensions
    {
        public static void Shuffle<T>(this IList<T> list)
        {
            RNGCryptoServiceProvider provider = new RNGCryptoServiceProvider();
            int n = list.Count;
            while (n > 1)
            {
                byte[] box = new byte[1];
                do provider.GetBytes(box); while (!(box[0] < n * (Byte.MaxValue / n)));
                int k = (box[0] % n);
                n--;
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static void AddRange<TKey, TValue>(
            this IDictionary<TKey, TValue> target,
            IDictionary<TKey, TValue> source, bool skipDuplicated = false)
        {
            if (target == null)
                throw new ArgumentNullException(nameof(target));
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            foreach (var element in source)
            {
                if (target.ContainsKey(element.Key))
                {
                    if (skipDuplicated) continue;

                    throw new Exception($"Adding duplicated key {element.Key}");
                }

                target.Add(element);
            }
        }
    }
}