use std::fmt::{self, Debug};

#[derive(Eq, PartialEq, Hash, Clone, Copy, PartialOrd, Ord)]
pub struct NonMaxU8 {
    repr: NonMaxU8Repr,
}

impl Debug for NonMaxU8 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.get())
    }
}

impl NonMaxU8 {
    const MAX: Self = NonMaxU8 {
        repr: NonMaxU8Repr::_248,
    };

    #[inline]
    pub fn new(n: usize) -> Option<Self> {
        if n > Self::MAX.get() as usize {
            None
        } else {
            let repr = unsafe { std::mem::transmute(n as u8) };
            Some(Self { repr })
        }
    }

    #[inline]
    pub fn get(self) -> u8 {
        unsafe { core::mem::transmute(self.repr) }
    }
}

macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

const_assert!(std::mem::size_of::<Option<NonMaxU8>>() == 1);

#[derive(Eq, PartialEq, Hash, Clone, Copy, PartialOrd, Ord)]
enum NonMaxU8Repr {
    _0 = 0,
    _1 = 1,
    _2 = 2,
    _3 = 3,
    _4 = 4,
    _5 = 5,
    _6 = 6,
    _7 = 7,
    _8 = 8,
    _9 = 9,
    _10 = 10,
    _11 = 11,
    _12 = 12,
    _13 = 13,
    _14 = 14,
    _15 = 15,
    _16 = 16,
    _17 = 17,
    _18 = 18,
    _19 = 19,
    _20 = 20,
    _21 = 21,
    _22 = 22,
    _23 = 23,
    _24 = 24,
    _25 = 25,
    _26 = 26,
    _27 = 27,
    _28 = 28,
    _29 = 29,
    _30 = 30,
    _31 = 31,
    _32 = 32,
    _33 = 33,
    _34 = 34,
    _35 = 35,
    _36 = 36,
    _37 = 37,
    _38 = 38,
    _39 = 39,
    _40 = 40,
    _41 = 41,
    _42 = 42,
    _43 = 43,
    _44 = 44,
    _45 = 45,
    _46 = 46,
    _47 = 47,
    _48 = 48,
    _49 = 49,
    _50 = 50,
    _51 = 51,
    _52 = 52,
    _53 = 53,
    _54 = 54,
    _55 = 55,
    _56 = 56,
    _57 = 57,
    _58 = 58,
    _59 = 59,
    _60 = 60,
    _61 = 61,
    _62 = 62,
    _63 = 63,
    _64 = 64,
    _65 = 65,
    _66 = 66,
    _67 = 67,
    _68 = 68,
    _69 = 69,
    _70 = 70,
    _71 = 71,
    _72 = 72,
    _73 = 73,
    _74 = 74,
    _75 = 75,
    _76 = 76,
    _77 = 77,
    _78 = 78,
    _79 = 79,
    _80 = 80,
    _81 = 81,
    _82 = 82,
    _83 = 83,
    _84 = 84,
    _85 = 85,
    _86 = 86,
    _87 = 87,
    _88 = 88,
    _89 = 89,
    _90 = 90,
    _91 = 91,
    _92 = 92,
    _93 = 93,
    _94 = 94,
    _95 = 95,
    _96 = 96,
    _97 = 97,
    _98 = 98,
    _99 = 99,
    _100 = 100,
    _101 = 101,
    _102 = 102,
    _103 = 103,
    _104 = 104,
    _105 = 105,
    _106 = 106,
    _107 = 107,
    _108 = 108,
    _109 = 109,
    _110 = 110,
    _111 = 111,
    _112 = 112,
    _113 = 113,
    _114 = 114,
    _115 = 115,
    _116 = 116,
    _117 = 117,
    _118 = 118,
    _119 = 119,
    _120 = 120,
    _121 = 121,
    _122 = 122,
    _123 = 123,
    _124 = 124,
    _125 = 125,
    _126 = 126,
    _127 = 127,
    _128 = 128,
    _129 = 129,
    _130 = 130,
    _131 = 131,
    _132 = 132,
    _133 = 133,
    _134 = 134,
    _135 = 135,
    _136 = 136,
    _137 = 137,
    _138 = 138,
    _139 = 139,
    _140 = 140,
    _141 = 141,
    _142 = 142,
    _143 = 143,
    _144 = 144,
    _145 = 145,
    _146 = 146,
    _147 = 147,
    _148 = 148,
    _149 = 149,
    _150 = 150,
    _151 = 151,
    _152 = 152,
    _153 = 153,
    _154 = 154,
    _155 = 155,
    _156 = 156,
    _157 = 157,
    _158 = 158,
    _159 = 159,
    _160 = 160,
    _161 = 161,
    _162 = 162,
    _163 = 163,
    _164 = 164,
    _165 = 165,
    _166 = 166,
    _167 = 167,
    _168 = 168,
    _169 = 169,
    _170 = 170,
    _171 = 171,
    _172 = 172,
    _173 = 173,
    _174 = 174,
    _175 = 175,
    _176 = 176,
    _177 = 177,
    _178 = 178,
    _179 = 179,
    _180 = 180,
    _181 = 181,
    _182 = 182,
    _183 = 183,
    _184 = 184,
    _185 = 185,
    _186 = 186,
    _187 = 187,
    _188 = 188,
    _189 = 189,
    _190 = 190,
    _191 = 191,
    _192 = 192,
    _193 = 193,
    _194 = 194,
    _195 = 195,
    _196 = 196,
    _197 = 197,
    _198 = 198,
    _199 = 199,
    _200 = 200,
    _201 = 201,
    _202 = 202,
    _203 = 203,
    _204 = 204,
    _205 = 205,
    _206 = 206,
    _207 = 207,
    _208 = 208,
    _209 = 209,
    _210 = 210,
    _211 = 211,
    _212 = 212,
    _213 = 213,
    _214 = 214,
    _215 = 215,
    _216 = 216,
    _217 = 217,
    _218 = 218,
    _219 = 219,
    _220 = 220,
    _221 = 221,
    _222 = 222,
    _223 = 223,
    _224 = 224,
    _225 = 225,
    _226 = 226,
    _227 = 227,
    _228 = 228,
    _229 = 229,
    _230 = 230,
    _231 = 231,
    _232 = 232,
    _233 = 233,
    _234 = 234,
    _235 = 235,
    _236 = 236,
    _237 = 237,
    _238 = 238,
    _239 = 239,
    _240 = 240,
    _241 = 241,
    _242 = 242,
    _243 = 243,
    _244 = 244,
    _245 = 245,
    _246 = 246,
    _247 = 247,
    _248 = 248,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all() {
        for i in 0..=(NonMaxU8::MAX.get() as usize) {
            let nm = NonMaxU8::new(i).unwrap();
            assert_eq!(i, nm.get() as usize);
            assert_eq!(format!("{:?}", i), format!("{:?}", nm));
        }
    }

    #[test]
    fn niche_rejected() {
        for i in (NonMaxU8::MAX.get() as usize + 1)..=(u8::MAX as usize) {
            assert_eq!(NonMaxU8::new(i), None);
        }
    }
}
