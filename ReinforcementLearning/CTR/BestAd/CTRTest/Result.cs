using FileHelpers;

namespace CTRTest
{
    [DelimitedRecord(";"), IgnoreFirst(1)]
    public class Result
    {
        public string Id { get; set; }
        public int YesNo { get; set; }
    }
}